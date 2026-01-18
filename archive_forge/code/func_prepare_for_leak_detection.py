from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def prepare_for_leak_detection() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    cmd_dict: T_JSON_DICT = {'method': 'Memory.prepareForLeakDetection'}
    json = (yield cmd_dict)