import warnings
from typing import Tuple
from urllib.parse import urlparse
from w3lib.url import safe_url_string
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.http import Request, Response
from scrapy.utils.misc import load_object
from scrapy.utils.python import to_unicode
from scrapy.utils.url import strip_url
def potentially_trustworthy(self, url):
    parsed_url = urlparse(url)
    if parsed_url.scheme in ('data',):
        return False
    return self.tls_protected(url)