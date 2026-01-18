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
def set_spidercls(self, url, opts):
    spider_loader = self.crawler_process.spider_loader
    if opts.spider:
        try:
            self.spidercls = spider_loader.load(opts.spider)
        except KeyError:
            logger.error('Unable to find spider: %(spider)s', {'spider': opts.spider})
    else:
        self.spidercls = spidercls_for_request(spider_loader, Request(url))
        if not self.spidercls:
            logger.error('Unable to find spider for: %(url)s', {'url': url})

    def _start_requests(spider):
        yield self.prepare_request(spider, Request(url), opts)
    if self.spidercls:
        self.spidercls.start_requests = _start_requests