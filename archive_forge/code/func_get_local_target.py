from abc import ABC, abstractmethod
from numba.core.registry import DelayedRegistry, CPUDispatcher
from numba.core.decorators import jit
from numba.core.errors import InternalTargetMismatchError, NumbaValueError
from threading import local as tls
def get_local_target(context):
    """
    Gets the local target from the call stack if available and the TLS
    override if not.
    """
    if len(context.callstack._stack) > 0:
        target = context.callstack[0].target
    else:
        target = target_registry.get(current_target(), None)
    if target is None:
        msg = 'The target found is not registered.Given target was {}.'
        raise ValueError(msg.format(target))
    else:
        return target