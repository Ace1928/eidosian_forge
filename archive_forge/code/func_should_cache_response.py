import gzip
import logging
import pickle
from email.utils import mktime_tz, parsedate_tz
from importlib import import_module
from pathlib import Path
from time import time
from weakref import WeakKeyDictionary
from w3lib.http import headers_dict_to_raw, headers_raw_to_dict
from scrapy.http import Headers, Response
from scrapy.http.request import Request
from scrapy.responsetypes import responsetypes
from scrapy.spiders import Spider
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.project import data_path
from scrapy.utils.python import to_bytes, to_unicode
def should_cache_response(self, response, request):
    cc = self._parse_cachecontrol(response)
    if b'no-store' in cc:
        return False
    if response.status == 304:
        return False
    if self.always_store:
        return True
    if b'max-age' in cc or b'Expires' in response.headers:
        return True
    if response.status in (300, 301, 308):
        return True
    if response.status in (200, 203, 401):
        return b'Last-Modified' in response.headers or b'ETag' in response.headers
    return False