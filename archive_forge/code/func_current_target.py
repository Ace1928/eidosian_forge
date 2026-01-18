from abc import ABC, abstractmethod
from numba.core.registry import DelayedRegistry, CPUDispatcher
from numba.core.decorators import jit
from numba.core.errors import InternalTargetMismatchError, NumbaValueError
from threading import local as tls
def current_target():
    """Returns the current target
    """
    return getattr(_active_context, 'target', _active_context_default)