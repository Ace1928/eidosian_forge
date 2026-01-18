from __future__ import annotations
import logging
import pprint
import signal
import warnings
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional, Set, Type, Union, cast
from twisted.internet.defer import (
from zope.interface.exceptions import DoesNotImplement
from zope.interface.verify import verifyClass
from scrapy import Spider, signals
from scrapy.addons import AddonManager
from scrapy.core.engine import ExecutionEngine
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.extension import ExtensionManager
from scrapy.interfaces import ISpiderLoader
from scrapy.logformatter import LogFormatter
from scrapy.settings import BaseSettings, Settings, overridden_settings
from scrapy.signalmanager import SignalManager
from scrapy.statscollectors import StatsCollector
from scrapy.utils.log import (
from scrapy.utils.misc import create_instance, load_object
from scrapy.utils.ossignal import install_shutdown_handlers, signal_names
from scrapy.utils.reactor import (
def create_crawler(self, crawler_or_spidercls: Union[Type[Spider], str, Crawler]) -> Crawler:
    """
        Return a :class:`~scrapy.crawler.Crawler` object.

        * If ``crawler_or_spidercls`` is a Crawler, it is returned as-is.
        * If ``crawler_or_spidercls`` is a Spider subclass, a new Crawler
          is constructed for it.
        * If ``crawler_or_spidercls`` is a string, this function finds
          a spider with this name in a Scrapy project (using spider loader),
          then creates a Crawler instance for it.
        """
    if isinstance(crawler_or_spidercls, Spider):
        raise ValueError('The crawler_or_spidercls argument cannot be a spider object, it must be a spider class (or a Crawler object)')
    if isinstance(crawler_or_spidercls, Crawler):
        return crawler_or_spidercls
    return self._create_crawler(crawler_or_spidercls)