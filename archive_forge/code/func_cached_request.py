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
def cached_request(self, request: PreparedRequest) -> HTTPResponse | Literal[False]:
    """
        Return a cached response if it exists in the cache, otherwise
        return False.
        """
    assert request.url is not None
    cache_url = self.cache_url(request.url)
    logger.debug('Looking up "%s" in the cache', cache_url)
    cc = self.parse_cache_control(request.headers)
    if 'no-cache' in cc:
        logger.debug('Request header has "no-cache", cache bypassed')
        return False
    if 'max-age' in cc and cc['max-age'] == 0:
        logger.debug('Request header has "max_age" as 0, cache bypassed')
        return False
    resp = self._load_from_cache(request)
    if not resp:
        return False
    if int(resp.status) in PERMANENT_REDIRECT_STATUSES:
        msg = 'Returning cached permanent redirect response (ignoring date and etag information)'
        logger.debug(msg)
        return resp
    headers: CaseInsensitiveDict[str] = CaseInsensitiveDict(resp.headers)
    if not headers or 'date' not in headers:
        if 'etag' not in headers:
            logger.debug('Purging cached response: no date or etag')
            self.cache.delete(cache_url)
        logger.debug('Ignoring cached response: no date')
        return False
    now = time.time()
    time_tuple = parsedate_tz(headers['date'])
    assert time_tuple is not None
    date = calendar.timegm(time_tuple[:6])
    current_age = max(0, now - date)
    logger.debug('Current age based on date: %i', current_age)
    resp_cc = self.parse_cache_control(headers)
    freshness_lifetime = 0
    max_age = resp_cc.get('max-age')
    if max_age is not None:
        freshness_lifetime = max_age
        logger.debug('Freshness lifetime from max-age: %i', freshness_lifetime)
    elif 'expires' in headers:
        expires = parsedate_tz(headers['expires'])
        if expires is not None:
            expire_time = calendar.timegm(expires[:6]) - date
            freshness_lifetime = max(0, expire_time)
            logger.debug('Freshness lifetime from expires: %i', freshness_lifetime)
    max_age = cc.get('max-age')
    if max_age is not None:
        freshness_lifetime = max_age
        logger.debug('Freshness lifetime from request max-age: %i', freshness_lifetime)
    min_fresh = cc.get('min-fresh')
    if min_fresh is not None:
        current_age += min_fresh
        logger.debug('Adjusted current age from min-fresh: %i', current_age)
    if freshness_lifetime > current_age:
        logger.debug('The response is "fresh", returning cached response')
        logger.debug('%i > %i', freshness_lifetime, current_age)
        return resp
    if 'etag' not in headers:
        logger.debug('The cached response is "stale" with no etag, purging')
        self.cache.delete(cache_url)
    return False