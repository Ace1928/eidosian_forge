from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def clear_messages() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Does nothing.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Console.clearMessages'}
    json = (yield cmd_dict)