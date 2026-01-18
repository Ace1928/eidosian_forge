from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def request_database_names(security_origin: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[str]]:
    """
    Requests database names for given security origin.

    :param security_origin: Security origin.
    :returns: Database names for origin.
    """
    params: T_JSON_DICT = dict()
    params['securityOrigin'] = security_origin
    cmd_dict: T_JSON_DICT = {'method': 'IndexedDB.requestDatabaseNames', 'params': params}
    json = (yield cmd_dict)
    return [str(i) for i in json['databaseNames']]