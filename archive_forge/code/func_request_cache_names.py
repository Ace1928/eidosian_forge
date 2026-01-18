from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def request_cache_names(security_origin: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[Cache]]:
    """
    Requests cache names.

    :param security_origin: Security origin.
    :returns: Caches for the security origin.
    """
    params: T_JSON_DICT = dict()
    params['securityOrigin'] = security_origin
    cmd_dict: T_JSON_DICT = {'method': 'CacheStorage.requestCacheNames', 'params': params}
    json = (yield cmd_dict)
    return [Cache.from_json(i) for i in json['caches']]