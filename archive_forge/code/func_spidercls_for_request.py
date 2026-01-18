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
def spidercls_for_request(spider_loader: SpiderLoader, request: Request, default_spidercls: Optional[Type[Spider]]=None, log_none: bool=False, log_multiple: bool=False) -> Optional[Type[Spider]]:
    """Return a spider class that handles the given Request.

    This will look for the spiders that can handle the given request (using
    the spider loader) and return a Spider class if (and only if) there is
    only one Spider able to handle the Request.

    If multiple spiders (or no spider) are found, it will return the
    default_spidercls passed. It can optionally log if multiple or no spiders
    are found.
    """
    snames = spider_loader.find_by_request(request)
    if len(snames) == 1:
        return spider_loader.load(snames[0])
    if len(snames) > 1 and log_multiple:
        logger.error('More than one spider can handle: %(request)s - %(snames)s', {'request': request, 'snames': ', '.join(snames)})
    if len(snames) == 0 and log_none:
        logger.error('Unable to find spider that handles: %(request)s', {'request': request})
    return default_spidercls