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
def get_callback_from_rules(self, spider, response):
    if getattr(spider, 'rules', None):
        for rule in spider.rules:
            if rule.link_extractor.matches(response.url):
                return rule.callback or 'parse'
    else:
        logger.error('No CrawlSpider rules found in spider %(spider)r, please specify a callback to use for parsing', {'spider': spider.name})