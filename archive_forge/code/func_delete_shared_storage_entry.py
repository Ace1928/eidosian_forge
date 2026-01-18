from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def delete_shared_storage_entry(owner_origin: str, key: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Deletes entry for ``key`` (if it exists) for a given origin's shared storage.

    **EXPERIMENTAL**

    :param owner_origin:
    :param key:
    """
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    params['key'] = key
    cmd_dict: T_JSON_DICT = {'method': 'Storage.deleteSharedStorageEntry', 'params': params}
    json = (yield cmd_dict)