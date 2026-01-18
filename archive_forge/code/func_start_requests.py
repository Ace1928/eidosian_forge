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
def start_requests(self) -> Iterable[Request]:
    if not self.start_urls and hasattr(self, 'start_url'):
        raise AttributeError("Crawling could not start: 'start_urls' not found or empty (but found 'start_url' attribute instead, did you miss an 's'?)")
    for url in self.start_urls:
        yield Request(url, dont_filter=True)