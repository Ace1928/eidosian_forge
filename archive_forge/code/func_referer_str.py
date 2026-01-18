import hashlib
import json
import warnings
from typing import (
from urllib.parse import urlunparse
from weakref import WeakKeyDictionary
from w3lib.http import basic_auth_header
from w3lib.url import canonicalize_url
from scrapy import Request, Spider
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.misc import load_object
from scrapy.utils.python import to_bytes, to_unicode
def referer_str(request: Request) -> Optional[str]:
    """Return Referer HTTP header suitable for logging."""
    referrer = request.headers.get('Referer')
    if referrer is None:
        return referrer
    return to_unicode(referrer, errors='replace')