from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def request_cached_response(cache_id: CacheId, request_url: str, request_headers: typing.List[Header]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, CachedResponse]:
    """
    Fetches cache entry.

    :param cache_id: Id of cache that contains the entry.
    :param request_url: URL spec of the request.
    :param request_headers: headers of the request.
    :returns: Response read from the cache.
    """
    params: T_JSON_DICT = dict()
    params['cacheId'] = cache_id.to_json()
    params['requestURL'] = request_url
    params['requestHeaders'] = [i.to_json() for i in request_headers]
    cmd_dict: T_JSON_DICT = {'method': 'CacheStorage.requestCachedResponse', 'params': params}
    json = (yield cmd_dict)
    return CachedResponse.from_json(json['response'])