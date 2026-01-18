from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def set_storage_bucket_tracking(storage_key: str, enable: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Set tracking for a storage key's buckets.

    **EXPERIMENTAL**

    :param storage_key:
    :param enable:
    """
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    params['enable'] = enable
    cmd_dict: T_JSON_DICT = {'method': 'Storage.setStorageBucketTracking', 'params': params}
    json = (yield cmd_dict)