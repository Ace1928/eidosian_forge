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
def cache_response(self, request: PreparedRequest, response: HTTPResponse, body: bytes | None=None, status_codes: Collection[int] | None=None) -> None:
    """
        Algorithm for caching requests.

        This assumes a requests Response object.
        """
    cacheable_status_codes = status_codes or self.cacheable_status_codes
    if response.status not in cacheable_status_codes:
        logger.debug('Status code %s not in %s', response.status, cacheable_status_codes)
        return
    response_headers: CaseInsensitiveDict[str] = CaseInsensitiveDict(response.headers)
    if 'date' in response_headers:
        time_tuple = parsedate_tz(response_headers['date'])
        assert time_tuple is not None
        date = calendar.timegm(time_tuple[:6])
    else:
        date = 0
    if body is not None and 'content-length' in response_headers and response_headers['content-length'].isdigit() and (int(response_headers['content-length']) != len(body)):
        return
    cc_req = self.parse_cache_control(request.headers)
    cc = self.parse_cache_control(response_headers)
    assert request.url is not None
    cache_url = self.cache_url(request.url)
    logger.debug('Updating cache with response from "%s"', cache_url)
    no_store = False
    if 'no-store' in cc:
        no_store = True
        logger.debug('Response header has "no-store"')
    if 'no-store' in cc_req:
        no_store = True
        logger.debug('Request header has "no-store"')
    if no_store and self.cache.get(cache_url):
        logger.debug('Purging existing cache entry to honor "no-store"')
        self.cache.delete(cache_url)
    if no_store:
        return
    if '*' in response_headers.get('vary', ''):
        logger.debug('Response header has "Vary: *"')
        return
    if self.cache_etags and 'etag' in response_headers:
        expires_time = 0
        if response_headers.get('expires'):
            expires = parsedate_tz(response_headers['expires'])
            if expires is not None:
                expires_time = calendar.timegm(expires[:6]) - date
        expires_time = max(expires_time, 14 * 86400)
        logger.debug(f'etag object cached for {expires_time} seconds')
        logger.debug('Caching due to etag')
        self._cache_set(cache_url, request, response, body, expires_time)
    elif int(response.status) in PERMANENT_REDIRECT_STATUSES:
        logger.debug('Caching permanent redirect')
        self._cache_set(cache_url, request, response, b'')
    elif 'date' in response_headers:
        time_tuple = parsedate_tz(response_headers['date'])
        assert time_tuple is not None
        date = calendar.timegm(time_tuple[:6])
        max_age = cc.get('max-age')
        if max_age is not None and max_age > 0:
            logger.debug('Caching b/c date exists and max-age > 0')
            expires_time = max_age
            self._cache_set(cache_url, request, response, body, expires_time)
        elif 'expires' in response_headers:
            if response_headers['expires']:
                expires = parsedate_tz(response_headers['expires'])
                if expires is not None:
                    expires_time = calendar.timegm(expires[:6]) - date
                else:
                    expires_time = None
                logger.debug('Caching b/c of expires header. expires in {} seconds'.format(expires_time))
                self._cache_set(cache_url, request, response, body, expires_time)