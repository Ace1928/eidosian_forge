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
def enqueue_scrape(self, result: Union[Response, Failure], request: Request, spider: Spider) -> Deferred:
    if self.slot is None:
        raise RuntimeError('Scraper slot not assigned')
    dfd = self.slot.add_response_request(result, request)

    def finish_scraping(_: Any) -> Any:
        assert self.slot is not None
        self.slot.finish_response(result, request)
        self._check_if_closing(spider)
        self._scrape_next(spider)
        return _
    dfd.addBoth(finish_scraping)
    dfd.addErrback(lambda f: logger.error('Scraper bug processing %(request)s', {'request': request}, exc_info=failure_to_exc_info(f), extra={'spider': spider}))
    self._scrape_next(spider)
    return dfd