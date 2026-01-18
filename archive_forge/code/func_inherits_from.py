from abc import ABC, abstractmethod
from numba.core.registry import DelayedRegistry, CPUDispatcher
from numba.core.decorators import jit
from numba.core.errors import InternalTargetMismatchError, NumbaValueError
from threading import local as tls
@classmethod
def inherits_from(cls, other):
    """Returns True if this target inherits from 'other' False otherwise"""
    return issubclass(cls, other)