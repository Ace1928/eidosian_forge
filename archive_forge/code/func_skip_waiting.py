from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target
def skip_waiting(scope_url: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    :param scope_url:
    """
    params: T_JSON_DICT = dict()
    params['scopeURL'] = scope_url
    cmd_dict: T_JSON_DICT = {'method': 'ServiceWorker.skipWaiting', 'params': params}
    json = (yield cmd_dict)