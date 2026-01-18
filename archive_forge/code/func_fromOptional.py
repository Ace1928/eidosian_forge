from functools import partial, wraps
from typing import Awaitable, Callable, Iterable, Optional, TypeVar
from twisted.internet.defer import Deferred, succeed
def fromOptional(default: _A, optional: Optional[_A]) -> _A:
    """
    Get a definite value from an optional value.

    @param default: The value to return if the optional value is missing.

    @param optional: The optional value to return if it exists.
    """
    if optional is None:
        return default
    return optional