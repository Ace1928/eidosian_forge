import logging
import re
import warnings
from scrapy import signals
from scrapy.http import Request
from scrapy.utils.httpobj import urlparse_cached
def should_follow(self, request, spider):
    regex = self.host_regex
    host = urlparse_cached(request).hostname or ''
    return bool(regex.search(host))