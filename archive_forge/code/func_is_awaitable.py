import inspect
from typing import Any
from types import CoroutineType, GeneratorType
def is_awaitable(value: Any) -> bool:
    """Return true if object can be passed to an ``await`` expression.

    Instead of testing if the object is an instance of abc.Awaitable, it checks
    the existence of an `__await__` attribute. This is much faster.
    """
    return isinstance(value, CoroutineType) or (isinstance(value, GeneratorType) and bool(value.gi_code.co_flags & CO_ITERABLE_COROUTINE)) or hasattr(value, '__await__')