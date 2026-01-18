from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def get_database_table_names(database_id: DatabaseId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[str]]:
    """
    :param database_id:
    :returns: 
    """
    params: T_JSON_DICT = dict()
    params['databaseId'] = database_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Database.getDatabaseTableNames', 'params': params}
    json = (yield cmd_dict)
    return [str(i) for i in json['tableNames']]