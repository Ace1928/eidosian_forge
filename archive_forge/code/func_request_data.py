from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def request_data(security_origin: str, database_name: str, object_store_name: str, index_name: str, skip_count: int, page_size: int, key_range: typing.Optional[KeyRange]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[typing.List[DataEntry], bool]]:
    """
    Requests data from object store or index.

    :param security_origin: Security origin.
    :param database_name: Database name.
    :param object_store_name: Object store name.
    :param index_name: Index name, empty string for object store data requests.
    :param skip_count: Number of records to skip.
    :param page_size: Number of records to fetch.
    :param key_range: *(Optional)* Key range.
    :returns: A tuple with the following items:

        0. **objectStoreDataEntries** - Array of object store data entries.
        1. **hasMore** - If true, there are more entries to fetch in the given range.
    """
    params: T_JSON_DICT = dict()
    params['securityOrigin'] = security_origin
    params['databaseName'] = database_name
    params['objectStoreName'] = object_store_name
    params['indexName'] = index_name
    params['skipCount'] = skip_count
    params['pageSize'] = page_size
    if key_range is not None:
        params['keyRange'] = key_range.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'IndexedDB.requestData', 'params': params}
    json = (yield cmd_dict)
    return ([DataEntry.from_json(i) for i in json['objectStoreDataEntries']], bool(json['hasMore']))