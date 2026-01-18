import inspect
import json
import logging
from typing import Dict
from itemadapter import ItemAdapter, is_item
from twisted.internet.defer import maybeDeferred
from w3lib.url import is_url
from scrapy.commands import BaseRunSpiderCommand
from scrapy.exceptions import UsageError
from scrapy.http import Request
from scrapy.utils import display
from scrapy.utils.asyncgen import collect_asyncgen
from scrapy.utils.defer import aiter_errback, deferred_from_coro
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import arg_to_iter
from scrapy.utils.spider import spidercls_for_request
def iterate_spider_output(self, result):
    if inspect.isasyncgen(result):
        d = deferred_from_coro(collect_asyncgen(aiter_errback(result, self.handle_exception)))
        d.addCallback(self.iterate_spider_output)
        return d
    if inspect.iscoroutine(result):
        d = deferred_from_coro(result)
        d.addCallback(self.iterate_spider_output)
        return d
    return arg_to_iter(deferred_from_coro(result))