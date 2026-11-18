import struct
import logging

import angr
from angr import claripy
from angr.calling_conventions import SimRegArg, SimStackArg

from .builder import Builder
from .. import rop_utils
from ..errors import RopException
from ..rop_gadget import FunctionGadget

l = logging.getLogger(__name__)

class FuncCaller(Builder):
    """
    handle function calls by automatically detecting the target binary's
    calling convention
    thanks to @tomgond for a great portion of this class
    """
    def __init__(self, chain_builder):
        super().__init__(chain_builder)
        # invoke a function but cannot maintain the control flow afterwards (pop rdi; jmp rax)
        self._func_jmp_gadgets = None
        # invoke a function and still maintain the control flow afterwards (call rax; ret)
        # TODO: currently not supported
        self._func_call_gadgets = None
        # cache of csu-epilogue discovery results, keyed by gadget address.
        # discovery is a pure function of (gadget, binary), so each gadget pays the
        # symbolic execution at most once per FuncCaller; None ("not a returning
        # dispatch") is cached too, so non-csu gadgets also pay the cost only once.
        self._csu_epilogue_cache = {}
        # record the calling convention
        self._cc = angr.default_cc(
                            self.project.arch.name,
                            platform=self.project.simos.name if self.project.simos is not None else None,
                            )(self.project.arch) # type:ignore

    def bootstrap(self):
        cc = self._cc
        self._func_jmp_gadgets = set()
        for g in self.chain_builder.gadgets:
            if g.self_contained:
                continue
            if g.popped_regs.intersection(cc.ARG_REGS):
                self._func_jmp_gadgets.add(g)
                continue
            for move in g.reg_moves:
                if move.to_reg in cc.ARG_REGS:
                    self._func_jmp_gadgets.add(g)
                    break

    def _find_function_pointer_in_got_plt(self, func_addr):
        """
        Search if a func addr is in plt. If it's in plt, find func name and
        translate it to GOT so that we can directly call/jmp to the location pointed there.
        """
        # Search GOT and PLT across all loaded objects
        func_name = None
        for sym in self.project.loader.main_object.symbols:
            if sym.rebased_addr == func_addr:
                func_name = sym.name

        for sym, val in self.project.loader.main_object.plt.items():
            if val == func_addr:
                func_name = sym
        # addr is found in plt. we look for this symbol in got
        if func_name:
            func_got = self.project.loader.main_object.imports.get(func_name)
            if func_got:
                return func_got.rebased_addr
            else:
                # this is from plt but not in got somehow
                return None
        # not in plt. We can search in other ways
        else:
            return None

    def _find_function_pointer(self, func_addr):
        """Find pointer to function, allowing for potential memory locations"""
        # Existing GOT/PLT search logic first
        got_ptr = self._find_function_pointer_in_got_plt(func_addr)
        if got_ptr is not None:
            return got_ptr

        # Broader search strategy
        func_ptr_bytes = struct.pack(self.project.arch.struct_fmt(), func_addr)
        for seg in self.project.loader.main_object.segments:
            if not seg.is_readable:
                continue
            if not seg.memsize:
                continue

            # Scan segments for potential pointers
            sec_data = self.project.loader.memory.load(seg.min_addr, seg.memsize)
            offset = sec_data.find(func_ptr_bytes)
            if offset == -1:
                continue
            return seg.min_addr + offset
        return None

    def _func_call(self, func_gadget, cc, args, extra_regs=None, preserve_regs=None,
                   needs_return=True, jmp_mem_target=None, **kwargs):
        """
        func_gadget: the address of the function to invoke
        cc: calling convention
        args: the arguments to the function
        extra_regs: what extra registers to set besides the function arguments, useful for invoking system calls
        preserve_res: what registers preserve
        needs_return: whether we need to cleanup stack after the function invocation,
            setting this to False will result in a shorter chain
        """
        assert type(args) in [list, tuple], "function arguments must be a list or tuple!"
        if kwargs:
            l.warning("passing deprecated arguments %s to angrop.chain_builder.FuncCaller", kwargs)

        preserve_regs = set(preserve_regs) if preserve_regs else set()
        arch_bytes = self.project.arch.bytes

        # distinguish register and stack arguments
        register_arguments = args
        stack_arguments = []
        if len(args) > len(cc.ARG_REGS):
            register_arguments = args[:len(cc.ARG_REGS)]
            stack_arguments = args[len(cc.ARG_REGS):]

        # set register arguments
        if needs_return and isinstance(cc.RETURN_ADDR, SimRegArg) and cc.RETURN_ADDR.reg_name != 'ip_at_syscall':
            reg_name = cc.RETURN_ADDR.reg_name
            preserve_regs.add(reg_name)
        registers = {} if extra_regs is None else extra_regs
        for arg, reg in zip(register_arguments, cc.ARG_REGS):
            registers[reg] = arg
        for reg in preserve_regs:
            registers.pop(reg, None)

        # if this is a simple function call, just set the registers and invoke it
        if not jmp_mem_target:
            chain = self.chain_builder.set_regs(**registers, preserve_regs=preserve_regs)
        else:
            # this is a jmp_mem function call, we need to constrain the jmp_mem target
            rop_values, constraints = self._build_ast_constraints(func_gadget.pc_target)
            registers.update(rop_values)
            chain = self.chain_builder.set_regs(**registers, preserve_regs=preserve_regs)
            state = chain._blank_state
            state.solver.add(claripy.And(*constraints))
            state.solver.add(jmp_mem_target == func_gadget.pc_target)
        # invoke the function
        chain.add_gadget(func_gadget)
        # we are done here if we don't need to return
        if not needs_return:
            return chain
        # recover stack from previous gadget effect.
        for delta in range(func_gadget.stack_change//arch_bytes):
            if func_gadget.pc_offset is None or delta != func_gadget.pc_offset:
                chain.add_value(self._get_fill_val())
            else:
                chain.add_value(claripy.BVS("next_pc", self.project.arch.bits))
        # now we need to cleanly finish the calling convention
        # 1. handle stack arguments
        # 2. handle function return address to maintain the control flow
        if stack_arguments:
            shift_bytes = (len(stack_arguments)+1)*arch_bytes
            # TODO: currently, we only shift stack only for the minimal
            # but if this shift fails, we should try larger shifts
            cleaner = self.chain_builder.shift(shift_bytes, next_pc_idx=-1, preserve_regs=preserve_regs)
            chain.add_gadget(cleaner._gadgets[0])
            for arg in stack_arguments:
                chain.add_value(arg)
            next_pc = claripy.BVS("next_pc", self.project.arch.bits)
            chain.add_value(next_pc)

        # handle return address
        if not isinstance(cc.RETURN_ADDR, (SimStackArg, SimRegArg)):
            raise RopException(f"What is the calling convention {cc} I'm dealing with?")
        if isinstance(cc.RETURN_ADDR, SimRegArg) and cc.RETURN_ADDR.reg_name != 'ip_at_syscall':
            # now we know this function will return to a specific register
            # so we need to set the return address before invoking the function
            reg_name = cc.RETURN_ADDR.reg_name
            shifter = self.chain_builder._shifter.shift(self.project.arch.bytes)
            next_ip = rop_utils.cast_rop_value(shifter._gadgets[0].addr, self.project)
            pre_chain = self.chain_builder.set_regs(**{reg_name: next_ip})
            chain = pre_chain + chain
        return chain

    def _parse_csu_dispatch(self, pc_target):
        """
        For a csu-style indirect call `call [base + idx*scale]`, return (base_reg, idx_reg).
        Returns None if pc_target is not of that shape.
        """
        if pc_target is None or pc_target.op != '__add__':
            return None
        base_reg = idx_reg = None
        for arg in pc_target.args:
            if arg.op == 'BVS' and arg.args[0].startswith('sreg_'):
                base_reg = arg.args[0][5:].split('-')[0]
            elif arg.op in ('__lshift__', '__mul__'):
                for sub in rop_utils.children_asts(arg):
                    if sub.op == 'BVS' and sub.args[0].startswith('sreg_'):
                        idx_reg = sub.args[0][5:].split('-')[0]
                        break
        if base_reg and idx_reg:
            return base_reg, idx_reg
        return None

    def _discover_csu_epilogue(self, g, base_reg, idx_reg):
        """
        B1: discover how a csu-style dispatch gadget returns after its `call [base+idx*8]`,
        by symbolic execution (no hardcoded glibc layout).

        We execute from g.addr with:
          - idx = 0, base = a scratch table whose entry 0 is a stub that returns at once,
          - every other register left symbolic,
          - the stack filled with distinct *symbolic* words.
        The csu epilogue then runs; where control finally `ret`s determines everything:
          - cleanup_slots = index of the stack word that becomes the (unconstrained) pc,
          - pins = the input-register values the solver *requires* to reach that clean
            return (e.g. the loop cap rbp==1 so the dispatch loop runs exactly once),
            excluding base_reg (set to the real function pointer later).

        Returns {'cleanup_slots': int, 'pins': {reg: val}} or None if no clean return
        path is found (then g is treated as non-returning).
        """
        # the returning-ret2csu mechanism is only proven on x86-64 (AMD64). the
        # pc_target parser CAN still match on other arches (e.g. ARM glibc 2.31 has
        # 4 such jmp_mem gadgets) and there the discovery symbolic execution stalls
        # for minutes (>11min CPU measured on one ARM gadget, vs 0.09s on AMD64),
        # so gate it off and let the caller fall back to a terminal call. See
        # INDEPENDENT-REVIEW-VERDICT.md (review finding C6-5).
        if self.project.arch.name != "AMD64":
            return None
        # per-gadget cache: repeated func_call()s must not re-pay the symbolic exec
        if g.addr in self._csu_epilogue_cache:
            return self._csu_epilogue_cache[g.addr]

        p = self.project
        STACK = 0x41410000
        TABLE = 0x42420000

        # a stub at FAKE that simply returns (pops the return address), so the indirect
        # call comes straight back into the epilogue.
        FAKE = 0x43430000
        p.hook(FAKE, angr.SIM_PROCEDURES['stubs']['ReturnUnconstrained'](), replace=True)
        try:
            result = self._discover_csu_epilogue_inner(g, base_reg, idx_reg, STACK, TABLE, FAKE)
        finally:
            p.unhook(FAKE)
        self._csu_epilogue_cache[g.addr] = result
        return result

    def _discover_csu_epilogue_inner(self, g, base_reg, idx_reg, STACK, TABLE, FAKE):
        import re
        p = self.project
        abytes = p.arch.bytes
        bits = p.arch.bits

        st = p.factory.blank_state(addr=g.addr, add_options={
            angr.options.SYMBOL_FILL_UNCONSTRAINED_MEMORY,
            angr.options.SYMBOL_FILL_UNCONSTRAINED_REGISTERS})
        end = p.arch.memory_endness
        setattr(st.regs, idx_reg, claripy.BVV(0, bits))
        setattr(st.regs, base_reg, claripy.BVV(TABLE, bits))
        st.memory.store(TABLE, claripy.BVV(FAKE, bits), endness=end)
        st.regs.sp = claripy.BVV(STACK, bits)

        # candidate input registers whose required values we will read back
        cand = {base_reg, idx_reg}
        for m in g.reg_moves:
            cand.add(m.from_reg)
        for r in ('rbx', 'rbp', 'r12', 'r13', 'r14', 'r15'):
            if hasattr(st.regs, r):
                cand.add(r)
        init_regs = {r: getattr(st.regs, r) for r in cand}

        # fill stack slots with *symbolic* words so a `ret` into any of them yields an
        # unconstrained pc we can trace back to its slot index.
        nslots = 32
        for i in range(nslots):
            st.memory.store(STACK + i*abytes,
                            claripy.BVS("csuslot_%d" % i, bits), endness=end)

        simgr = p.factory.simulation_manager(st, save_unconstrained=True)
        for _ in range(200):
            if not (simgr.active or simgr.unconstrained):
                break
            for u in simgr.unconstrained:
                names = [v for v in u.regs.pc.variables if v.startswith("csuslot_")]
                if len(names) != 1:
                    continue
                cleanup_slots = int(re.match(r"csuslot_(\d+)", names[0]).group(1))
                pins = {}
                for r, sym in init_regs.items():
                    if r == base_reg:
                        continue
                    vals = u.solver.eval_upto(sym, 2)
                    if len(vals) == 1:
                        pins[r] = vals[0]
                return {'cleanup_slots': int(cleanup_slots), 'pins': pins}
            simgr.step()
        return None

    def _try_csu_return(self, g, registers, ptr_to_func):
        """
        Blocker B: assemble a RETURNING ret2csu through a csu-style dispatch gadget g
        (`mov args->arg_regs; call [base+idx*8]`). Returns a RopChain, or None if g is
        not a returning csu dispatch (caller then falls back to a terminal call).
        """
        parsed = self._parse_csu_dispatch(getattr(g, 'pc_target', None))
        if parsed is None:
            return None
        base_reg, idx_reg = parsed
        epi = self._discover_csu_epilogue(g, base_reg, idx_reg)
        if epi is None:
            return None
        cleanup_slots = epi['cleanup_slots']

        # B3: pin the loop-control / table-index regs that discovery found are required
        # for the epilogue to return cleanly (e.g. idx=0, cap=1). base_reg is excluded by
        # discovery; for the single-entry case (idx=0) it must equal the function pointer.
        extra = dict(registers)
        extra.update(epi['pins'])
        extra[base_reg] = ptr_to_func

        # B2: carry the epilogue on a copy of the gadget so _func_call's existing return
        # loop lays out [fill]*cleanup_slots + next_pc (see func_caller return handling).
        # MUST use RopGadget.copy(), NOT copy.copy(): copy.copy routes through
        # __getstate__ (which nulls `project` for pickling), leaving the synthesized
        # gadget with project=None — then str(chain)/dstr()/payload_code() raise
        # AttributeError on every returning chain (independent-review finding NF1).
        g2 = g.copy()
        g2.stack_change = (cleanup_slots + 1) * self.project.arch.bytes
        g2.pc_offset = cleanup_slots
        try:
            return self._func_call(g2, self._cc, [], extra_regs=extra, needs_return=True)
        except RopException:
            return None

    def func_call(self, address, args, **kwargs):
        """
        :param address: address or name of function to call
        :param args: a list/tuple of arguments to the function
        :param preserve_regs: list of registers which shouldn't be set
        :param needs_return: whether to continue the ROP after invoking the function
        :return: a RopChain which invokes the function with the arguments
        """
        symbol = None
        # is it a symbol?
        if isinstance(address, str):
            symbol = address
            symobj = self.project.loader.main_object.get_symbol(symbol)
            if hasattr(self.project.loader.main_object, 'plt') and address in self.project.loader.main_object.plt:
                address = self.project.loader.main_object.plt[symbol]
            elif symobj is not None:
                address = symobj.rebased_addr
            else:
                raise RopException("Symbol passed to func_call does not exist in the binary")

        # try to invoke the function using all self-contained gadgets
        func_gadget = FunctionGadget(address, symbol)
        func_gadget.stack_change = self.project.arch.bytes
        func_gadget.pc_offset = 0
        # Synthesized gadgets never pass through _create_gadget, so tag has_endbr here;
        # otherwise a jmp_reg -> func transition would be spuriously rejected under IBT.
        if self.arch.ibt or self.arch.force_endbr:
            func_gadget.has_endbr = self.arch.addr_has_endbr(address)
        try:
            return self._func_call(func_gadget, self._cc, args, **kwargs)
        except RopException:
            pass

        # well, let's try non-self-contained gadgets.
        # a csu-style dispatch (`mov args; call [base+idx*8]`) returns through its
        # epilogue, so for those we CAN still honor needs_return; plain jmp gadgets
        # (e.g. `pop rdi; jmp rax`) cannot, and keep the old fire-and-forget behavior.
        want_return = kwargs.get("needs_return", None)   # True / False / None

        # try func_jmp_gadgets
        register_args = args[:len(self._cc.ARG_REGS)]
        registers = {self._cc.ARG_REGS[i]:register_args[i] for i in range(len(register_args))}
        reg_names = set(registers.keys())
        ptr_to_func = self._find_function_pointer(address)
        hard_regs = [x for x in registers if not self.chain_builder._reg_setter.can_set_reg(x)]
        if ptr_to_func is not None:
            for g in self._func_jmp_gadgets: # type:ignore
                if g.popped_regs.intersection(reg_names):
                    l.warning("do not support func_jmp_gadgets that have pops: %s", g.dstr())
                    continue

                # build the new target registers
                registers = registers.copy()
                skip = False
                # RELAXED GUARD (Blocker A): keep a multi-move gadget as long as it
                # supplies at least one otherwise-unsettable ("hard") arg register.
                # The csu dispatch also moves into rdi/rsi (settable), which the old
                # `move.to_reg not in hard_regs` test rejected wholesale.
                if not any(m.to_reg in hard_regs for m in g.reg_moves):
                    continue
                for move in g.reg_moves:
                    if move.from_reg in hard_regs:
                        skip = True
                        break
                    if move.to_reg in registers.keys():
                        val = registers[move.to_reg]
                        if move.from_reg in registers:
                            l.warning("oops, overlapped moves not handled atm: %s", g.dstr())
                            skip = True
                            break
                        del registers[move.to_reg]
                        registers[move.from_reg] = val
                if skip:
                    continue

                if g.transit_type != 'jmp_mem':
                    raise NotImplementedError("currently only support jmp_mem type func_jmp_gadgets!")

                # Blocker B: try a RETURNING csu-style call first.
                if want_return is not False:
                    csu = self._try_csu_return(g, registers.copy(), ptr_to_func)
                    if csu is not None:
                        return csu
                    if want_return:
                        # caller demanded a return but this gadget cannot provide one
                        continue

                # fall back to the original terminal (non-returning) jmp_mem call
                if want_return is None:
                    l.warning("function %s won't return!", symbol if symbol else hex(address))
                kw = dict(kwargs)
                kw['needs_return'] = False
                try:
                    return self._func_call(g, self._cc, [], extra_regs=registers,
                                           jmp_mem_target=ptr_to_func, **kw)
                except RopException:
                    pass

        s = symbol if symbol else hex(address)
        raise RopException(f"fail to invoke function: {s}")
