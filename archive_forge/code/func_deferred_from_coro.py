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
def deferred_from_coro(o: _T) -> Union[Deferred, _T]:
    """Converts a coroutine into a Deferred, or returns the object as is if it isn't a coroutine"""
    if isinstance(o, Deferred):
        return o
    if asyncio.isfuture(o) or inspect.isawaitable(o):
        if not is_asyncio_reactor_installed():
            return ensureDeferred(cast(Coroutine[Deferred, Any, Any], o))
        event_loop = _get_asyncio_event_loop()
        return Deferred.fromFuture(asyncio.ensure_future(o, loop=event_loop))
    return o