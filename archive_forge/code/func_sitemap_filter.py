import logging
import re
from typing import TYPE_CHECKING, Any
from scrapy.http import Request, XmlResponse
from scrapy.spiders import Spider
from scrapy.utils._compression import _DecompressionMaxSizeExceeded
from scrapy.utils.gz import gunzip, gzip_magic_number
from scrapy.utils.sitemap import Sitemap, sitemap_urls_from_robots
def sitemap_filter(self, entries):
    """This method can be used to filter sitemap entries by their
        attributes, for example, you can filter locs with lastmod greater
        than a given date (see docs).
        """
    for entry in entries:
        yield entry