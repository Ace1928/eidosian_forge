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
def request_to_curl(request: Request) -> str:
    """
    Converts a :class:`~scrapy.Request` object to a curl command.

    :param :class:`~scrapy.Request`: Request object to be converted
    :return: string containing the curl command
    """
    method = request.method
    data = f"--data-raw '{request.body.decode('utf-8')}'" if request.body else ''
    headers = ' '.join((f"-H '{k.decode()}: {v[0].decode()}'" for k, v in request.headers.items()))
    url = request.url
    cookies = ''
    if request.cookies:
        if isinstance(request.cookies, dict):
            cookie = '; '.join((f'{k}={v}' for k, v in request.cookies.items()))
            cookies = f"--cookie '{cookie}'"
        elif isinstance(request.cookies, list):
            cookie = '; '.join((f'{list(c.keys())[0]}={list(c.values())[0]}' for c in request.cookies))
            cookies = f"--cookie '{cookie}'"
    curl_cmd = f'curl -X {method} {url} {data} {headers} {cookies}'.strip()
    return ' '.join(curl_cmd.split())