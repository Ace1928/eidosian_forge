from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def delete_object_store_entries(security_origin: str, database_name: str, object_store_name: str, key_range: KeyRange) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Delete a range of entries from an object store

    :param security_origin:
    :param database_name:
    :param object_store_name:
    :param key_range: Range of entry keys to delete
    """
    params: T_JSON_DICT = dict()
    params['securityOrigin'] = security_origin
    params['databaseName'] = database_name
    params['objectStoreName'] = object_store_name
    params['keyRange'] = key_range.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'IndexedDB.deleteObjectStoreEntries', 'params': params}
    json = (yield cmd_dict)