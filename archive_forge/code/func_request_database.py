from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def request_database(security_origin: str, database_name: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, DatabaseWithObjectStores]:
    """
    Requests database with given name in given frame.

    :param security_origin: Security origin.
    :param database_name: Database name.
    :returns: Database with an array of object stores.
    """
    params: T_JSON_DICT = dict()
    params['securityOrigin'] = security_origin
    params['databaseName'] = database_name
    cmd_dict: T_JSON_DICT = {'method': 'IndexedDB.requestDatabase', 'params': params}
    json = (yield cmd_dict)
    return DatabaseWithObjectStores.from_json(json['databaseWithObjectStores'])