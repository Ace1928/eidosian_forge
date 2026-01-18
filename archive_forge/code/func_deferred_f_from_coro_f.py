import asyncio
import inspect
from asyncio import Future
from functools import wraps
from types import CoroutineType
from typing import (
from twisted.internet import defer
from twisted.internet.defer import Deferred, DeferredList, ensureDeferred
from twisted.internet.task import Cooperator
from twisted.python import failure
from twisted.python.failure import Failure
from scrapy.exceptions import IgnoreRequest
from scrapy.utils.reactor import _get_asyncio_event_loop, is_asyncio_reactor_installed
def deferred_f_from_coro_f(coro_f: Callable[..., Coroutine]) -> Callable:
    """Converts a coroutine function into a function that returns a Deferred.

    The coroutine function will be called at the time when the wrapper is called. Wrapper args will be passed to it.
    This is useful for callback chains, as callback functions are called with the previous callback result.
    """

    @wraps(coro_f)
    def f(*coro_args: Any, **coro_kwargs: Any) -> Any:
        return deferred_from_coro(coro_f(*coro_args, **coro_kwargs))
    return f