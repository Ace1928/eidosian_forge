from __future__ import annotations
import logging
from collections import deque
from typing import (
from itemadapter import is_item
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure
from scrapy import Spider, signals
from scrapy.core.spidermw import SpiderMiddlewareManager
from scrapy.exceptions import CloseSpider, DropItem, IgnoreRequest
from scrapy.http import Request, Response
from scrapy.logformatter import LogFormatter
from scrapy.pipelines import ItemPipelineManager
from scrapy.signalmanager import SignalManager
from scrapy.utils.defer import (
from scrapy.utils.log import failure_to_exc_info, logformatter_adapter
from scrapy.utils.misc import load_object, warn_on_generator_with_return_value
from scrapy.utils.spider import iterate_spider_output
def handle_spider_output(self, result: Union[Iterable, AsyncIterable], request: Request, response: Union[Response, Failure], spider: Spider) -> Deferred:
    if not result:
        return defer_succeed(None)
    it: Union[Generator, AsyncGenerator]
    if isinstance(result, AsyncIterable):
        it = aiter_errback(result, self.handle_spider_error, request, response, spider)
        dfd = parallel_async(it, self.concurrent_items, self._process_spidermw_output, request, response, spider)
    else:
        it = iter_errback(result, self.handle_spider_error, request, response, spider)
        dfd = parallel(it, self.concurrent_items, self._process_spidermw_output, request, response, spider)
    return dfd