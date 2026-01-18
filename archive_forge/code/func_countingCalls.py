from functools import partial, wraps
from typing import Awaitable, Callable, Iterable, Optional, TypeVar
from twisted.internet.defer import Deferred, succeed
def countingCalls(f: Callable[[int], _A]) -> Callable[[], _A]:
    """
    Wrap a function with another that automatically passes an integer counter
    of the number of calls that have gone through the wrapper.
    """
    counter = 0

    def g() -> _A:
        nonlocal counter
        try:
            result = f(counter)
        finally:
            counter += 1
        return result
    return g