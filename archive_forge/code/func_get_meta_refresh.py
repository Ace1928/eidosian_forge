import os
import re
import tempfile
import webbrowser
from typing import Any, Callable, Iterable, Tuple, Union
from weakref import WeakKeyDictionary
from twisted.web import http
from w3lib import html
import scrapy
from scrapy.http.response import Response
from scrapy.utils.decorators import deprecated
from scrapy.utils.python import to_bytes, to_unicode
def get_meta_refresh(response: 'scrapy.http.response.text.TextResponse', ignore_tags: Iterable[str]=('script', 'noscript')) -> Union[Tuple[None, None], Tuple[float, str]]:
    """Parse the http-equiv refresh parameter from the given response"""
    if response not in _metaref_cache:
        text = response.text[0:4096]
        _metaref_cache[response] = html.get_meta_refresh(text, response.url, response.encoding, ignore_tags=ignore_tags)
    return _metaref_cache[response]