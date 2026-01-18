from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def get_isolate_id() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, str]:
    """
    Returns the isolate id.

    **EXPERIMENTAL**

    :returns: The isolate id.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.getIsolateId'}
    json = (yield cmd_dict)
    return str(json['id'])