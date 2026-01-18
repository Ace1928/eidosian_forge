from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def clear_data_for_storage_key(storage_key: str, storage_types: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Clears storage for storage key.

    :param storage_key: Storage key.
    :param storage_types: Comma separated list of StorageType to clear.
    """
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    params['storageTypes'] = storage_types
    cmd_dict: T_JSON_DICT = {'method': 'Storage.clearDataForStorageKey', 'params': params}
    json = (yield cmd_dict)