from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def get_shared_storage_entries(owner_origin: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[SharedStorageEntry]]:
    """
    Gets the entries in an given origin's shared storage.

    **EXPERIMENTAL**

    :param owner_origin:
    :returns: 
    """
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    cmd_dict: T_JSON_DICT = {'method': 'Storage.getSharedStorageEntries', 'params': params}
    json = (yield cmd_dict)
    return [SharedStorageEntry.from_json(i) for i in json['entries']]