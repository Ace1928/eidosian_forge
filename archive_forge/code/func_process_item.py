from typing import Any, List
from twisted.internet.defer import Deferred
from scrapy import Spider
from scrapy.middleware import MiddlewareManager
from scrapy.utils.conf import build_component_list
from scrapy.utils.defer import deferred_f_from_coro_f
def process_item(self, item: Any, spider: Spider) -> Deferred:
    return self._process_chain('process_item', item, spider)