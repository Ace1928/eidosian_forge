from __future__ import annotations
import inspect
import logging
from types import CoroutineType, ModuleType
from typing import (
from twisted.internet.defer import Deferred
from scrapy import Request
from scrapy.spiders import Spider
from scrapy.utils.defer import deferred_from_coro
from scrapy.utils.misc import arg_to_iter
def iter_spider_classes(module: ModuleType) -> Generator[Type[Spider], Any, None]:
    """Return an iterator over all spider classes defined in the given module
    that can be instantiated (i.e. which have name)
    """
    from scrapy.spiders import Spider
    for obj in vars(module).values():
        if inspect.isclass(obj) and issubclass(obj, Spider) and (obj.__module__ == module.__name__) and getattr(obj, 'name', None):
            yield obj