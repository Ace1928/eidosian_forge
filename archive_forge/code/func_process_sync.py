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
def process_sync(iterable: Iterable) -> Generator:
    try:
        for r in iterable:
            yield r
    except Exception as ex:
        exception_result = self._process_spider_exception(response, spider, Failure(ex), exception_processor_index)
        if isinstance(exception_result, Failure):
            raise
        recover_to.extend(exception_result)