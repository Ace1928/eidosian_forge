from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def release_object_group(object_group: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Releases all remote objects that belong to a given group.

    :param object_group: Symbolic object group name.
    """
    params: T_JSON_DICT = dict()
    params['objectGroup'] = object_group
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.releaseObjectGroup', 'params': params}
    json = (yield cmd_dict)