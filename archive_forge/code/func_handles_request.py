from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Union, cast
from twisted.internet.defer import Deferred
from scrapy import signals
from scrapy.http import Request, Response
from scrapy.utils.trackref import object_ref
from scrapy.utils.url import url_is_from_spider
from scrapy.spiders.crawl import CrawlSpider, Rule
from scrapy.spiders.feed import CSVFeedSpider, XMLFeedSpider
from scrapy.spiders.sitemap import SitemapSpider
@classmethod
def handles_request(cls, request: Request) -> bool:
    return url_is_from_spider(request.url, cls)