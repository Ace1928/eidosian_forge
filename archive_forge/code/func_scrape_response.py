import logging
from inspect import isasyncgenfunction, iscoroutine
from itertools import islice
from typing import (
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure
from scrapy import Request, Spider
from scrapy.exceptions import _InvalidOutput
from scrapy.http import Response
from scrapy.middleware import MiddlewareManager
from scrapy.settings import BaseSettings
from scrapy.utils.asyncgen import as_async_generator, collect_asyncgen
from scrapy.utils.conf import build_component_list
from scrapy.utils.defer import (
from scrapy.utils.python import MutableAsyncChain, MutableChain
def scrape_response(self, scrape_func: ScrapeFunc, response: Response, request: Request, spider: Spider) -> Deferred:

    async def process_callback_output(result: Union[Iterable, AsyncIterable]) -> Union[MutableChain, MutableAsyncChain]:
        return await self._process_callback_output(response, spider, result)

    def process_spider_exception(_failure: Failure) -> Union[Failure, MutableChain]:
        return self._process_spider_exception(response, spider, _failure)
    dfd = mustbe_deferred(self._process_spider_input, scrape_func, response, request, spider)
    dfd.addCallbacks(callback=deferred_f_from_coro_f(process_callback_output), errback=process_spider_exception)
    return dfd