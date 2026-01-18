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
def defer_succeed(result: Any) -> Deferred:
    """Same as twisted.internet.defer.succeed but delay calling callback until
    next reactor loop

    It delays by 100ms so reactor has a chance to go through readers and writers
    before attending pending delayed calls, so do not set delay to zero.
    """
    from twisted.internet import reactor
    d: Deferred = Deferred()
    reactor.callLater(0.1, d.callback, result)
    return d