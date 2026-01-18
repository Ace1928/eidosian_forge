from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def set_shared_storage_entry(owner_origin: str, key: str, value: str, ignore_if_present: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets entry with ``key`` and ``value`` for a given origin's shared storage.

    **EXPERIMENTAL**

    :param owner_origin:
    :param key:
    :param value:
    :param ignore_if_present: *(Optional)* If ```ignoreIfPresent```` is included and true, then only sets the entry if ````key``` doesn't already exist.
    """
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    params['key'] = key
    params['value'] = value
    if ignore_if_present is not None:
        params['ignoreIfPresent'] = ignore_if_present
    cmd_dict: T_JSON_DICT = {'method': 'Storage.setSharedStorageEntry', 'params': params}
    json = (yield cmd_dict)