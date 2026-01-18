import warnings
from logging import getLogger
from scrapy import signals
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import Response, TextResponse
from scrapy.responsetypes import responsetypes
from scrapy.utils._compression import (
from scrapy.utils.deprecate import ScrapyDeprecationWarning
from scrapy.utils.gz import gunzip
def open_spider(self, spider):
    if hasattr(spider, 'download_maxsize'):
        self._max_size = spider.download_maxsize
    if hasattr(spider, 'download_warnsize'):
        self._warn_size = spider.download_warnsize