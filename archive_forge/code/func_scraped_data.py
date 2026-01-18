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
def scraped_data(self, args):
    items, requests, opts, depth, spider, callback = args
    if opts.pipelines:
        itemproc = self.pcrawler.engine.scraper.itemproc
        for item in items:
            itemproc.process_item(item, spider)
    self.add_items(depth, items)
    self.add_requests(depth, requests)
    scraped_data = items if opts.output else []
    if depth < opts.depth:
        for req in requests:
            req.meta['_depth'] = depth + 1
            req.meta['_callback'] = req.callback
            req.callback = callback
        scraped_data += requests
    return scraped_data