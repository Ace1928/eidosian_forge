import sys
from enum import IntFlag
from _pydevd_frame_eval.vendored import bytecode as _bytecode
def infer_flags(bytecode, is_async=None):
    """Infer the proper flags for a bytecode based on the instructions.

    Because the bytecode does not have enough context to guess if a function
    is asynchronous the algorithm tries to be conservative and will never turn
    a previously async code into a sync one.

    Parameters
    ----------
    bytecode : Bytecode | ConcreteBytecode | ControlFlowGraph
        Bytecode for which to infer the proper flags
    is_async : bool | None, optional
        Force the code to be marked as asynchronous if True, prevent it from
        being marked as asynchronous if False and simply infer the best
        solution based on the opcode and the existing flag if None.

    """
    flags = CompilerFlags(0)
    if not isinstance(bytecode, (_bytecode.Bytecode, _bytecode.ConcreteBytecode, _bytecode.ControlFlowGraph)):
        msg = 'Expected a Bytecode, ConcreteBytecode or ControlFlowGraph instance not %s'
        raise ValueError(msg % bytecode)
    instructions = bytecode.get_instructions() if isinstance(bytecode, _bytecode.ControlFlowGraph) else bytecode
    instr_names = {i.name for i in instructions if not isinstance(i, (_bytecode.SetLineno, _bytecode.Label))}
    if not instr_names & {'STORE_NAME', 'LOAD_NAME', 'DELETE_NAME'}:
        flags |= CompilerFlags.OPTIMIZED
    if not instr_names & {'LOAD_CLOSURE', 'LOAD_DEREF', 'STORE_DEREF', 'DELETE_DEREF', 'LOAD_CLASSDEREF'}:
        flags |= CompilerFlags.NOFREE
    flags |= bytecode.flags & (CompilerFlags.NEWLOCALS | CompilerFlags.VARARGS | CompilerFlags.VARKEYWORDS | CompilerFlags.NESTED)
    sure_generator = instr_names & {'YIELD_VALUE'}
    maybe_generator = instr_names & {'YIELD_VALUE', 'YIELD_FROM'}
    sure_async = instr_names & {'GET_AWAITABLE', 'GET_AITER', 'GET_ANEXT', 'BEFORE_ASYNC_WITH', 'SETUP_ASYNC_WITH', 'END_ASYNC_FOR'}
    if is_async in (None, True):
        if bytecode.flags & CompilerFlags.COROUTINE:
            if sure_generator:
                flags |= CompilerFlags.ASYNC_GENERATOR
            else:
                flags |= CompilerFlags.COROUTINE
        elif bytecode.flags & CompilerFlags.ITERABLE_COROUTINE:
            if sure_async:
                msg = 'The ITERABLE_COROUTINE flag is set but bytecode thatcan only be used in async functions have been detected. Please unset that flag before performing inference.'
                raise ValueError(msg)
            flags |= CompilerFlags.ITERABLE_COROUTINE
        elif bytecode.flags & CompilerFlags.ASYNC_GENERATOR:
            if not sure_generator:
                flags |= CompilerFlags.COROUTINE
            else:
                flags |= CompilerFlags.ASYNC_GENERATOR
        elif sure_async:
            if sure_generator:
                flags |= CompilerFlags.ASYNC_GENERATOR
            else:
                flags |= CompilerFlags.COROUTINE
        elif maybe_generator:
            if is_async:
                if sure_generator:
                    flags |= CompilerFlags.ASYNC_GENERATOR
                else:
                    flags |= CompilerFlags.COROUTINE
            else:
                flags |= CompilerFlags.GENERATOR
        elif is_async:
            flags |= CompilerFlags.COROUTINE
    else:
        if sure_async:
            raise ValueError('The is_async argument is False but bytecodes that can only be used in async functions have been detected.')
        if maybe_generator:
            flags |= CompilerFlags.GENERATOR
    flags |= bytecode.flags & CompilerFlags.FUTURE_GENERATOR_STOP
    return flags