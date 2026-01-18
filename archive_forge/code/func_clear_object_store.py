from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def clear_object_store(security_origin: str, database_name: str, object_store_name: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Clears all entries from an object store.

    :param security_origin: Security origin.
    :param database_name: Database name.
    :param object_store_name: Object store name.
    """
    params: T_JSON_DICT = dict()
    params['securityOrigin'] = security_origin
    params['databaseName'] = database_name
    params['objectStoreName'] = object_store_name
    cmd_dict: T_JSON_DICT = {'method': 'IndexedDB.clearObjectStore', 'params': params}
    json = (yield cmd_dict)