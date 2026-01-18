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
def maybe_deferred_to_future(d: Deferred) -> Union[Deferred, Future]:
    """
    .. versionadded:: 2.6.0

    Return *d* as an object that can be awaited from a :ref:`Scrapy callable
    defined as a coroutine <coroutine-support>`.

    What you can await in Scrapy callables defined as coroutines depends on the
    value of :setting:`TWISTED_REACTOR`:

    -   When not using the asyncio reactor, you can only await on
        :class:`~twisted.internet.defer.Deferred` objects.

    -   When :ref:`using the asyncio reactor <install-asyncio>`, you can only
        await on :class:`asyncio.Future` objects.

    If you want to write code that uses ``Deferred`` objects but works with any
    reactor, use this function on all ``Deferred`` objects::

        class MySpider(Spider):
            ...
            async def parse(self, response):
                additional_request = scrapy.Request('https://example.org/price')
                deferred = self.crawler.engine.download(additional_request)
                additional_response = await maybe_deferred_to_future(deferred)
    """
    if not is_asyncio_reactor_installed():
        return d
    return deferred_to_future(d)