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
def request_authenticate(request: Request, username: str, password: str) -> None:
    """Authenticate the given request (in place) using the HTTP basic access
    authentication mechanism (RFC 2617) and the given username and password
    """
    request.headers['Authorization'] = basic_auth_header(username, password)