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
def print_items(self, lvl=None, colour=True):
    if lvl is None:
        items = [item for lst in self.items.values() for item in lst]
    else:
        items = self.items.get(lvl, [])
    print('# Scraped Items ', '-' * 60)
    display.pprint([ItemAdapter(x).asdict() for x in items], colorize=colour)