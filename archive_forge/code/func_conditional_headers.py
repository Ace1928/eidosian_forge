from __future__ import annotations
import calendar
import logging
import re
import time
from email.utils import parsedate_tz
from typing import TYPE_CHECKING, Collection, Mapping
from pip._vendor.requests.structures import CaseInsensitiveDict
from pip._vendor.cachecontrol.cache import DictCache, SeparateBodyBaseCache
from pip._vendor.cachecontrol.serialize import Serializer
def conditional_headers(self, request: PreparedRequest) -> dict[str, str]:
    resp = self._load_from_cache(request)
    new_headers = {}
    if resp:
        headers: CaseInsensitiveDict[str] = CaseInsensitiveDict(resp.headers)
        if 'etag' in headers:
            new_headers['If-None-Match'] = headers['ETag']
        if 'last-modified' in headers:
            new_headers['If-Modified-Since'] = headers['Last-Modified']
    return new_headers