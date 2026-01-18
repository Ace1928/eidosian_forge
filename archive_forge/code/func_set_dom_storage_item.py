from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def set_dom_storage_item(storage_id: StorageId, key: str, value: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    :param storage_id:
    :param key:
    :param value:
    """
    params: T_JSON_DICT = dict()
    params['storageId'] = storage_id.to_json()
    params['key'] = key
    params['value'] = value
    cmd_dict: T_JSON_DICT = {'method': 'DOMStorage.setDOMStorageItem', 'params': params}
    json = (yield cmd_dict)