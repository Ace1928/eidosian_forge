import functools
import logging
from collections import defaultdict
from inspect import signature
from warnings import warn
from twisted.internet.defer import Deferred, DeferredList
from twisted.python.failure import Failure
from scrapy.http.request import NO_CALLBACK
from scrapy.settings import Settings
from scrapy.utils.datatypes import SequenceExclude
from scrapy.utils.defer import defer_result, mustbe_deferred
from scrapy.utils.deprecate import ScrapyDeprecationWarning
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import arg_to_iter
def item_completed(self, results, item, info):
    """Called per item when all media requests has been processed"""
    if self.LOG_FAILED_RESULTS:
        for ok, value in results:
            if not ok:
                logger.error('%(class)s found errors processing %(item)s', {'class': self.__class__.__name__, 'item': item}, exc_info=failure_to_exc_info(value), extra={'spider': info.spider})
    return item