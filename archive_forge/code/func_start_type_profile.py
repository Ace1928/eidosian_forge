from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
def start_type_profile() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enable type profile.

    **EXPERIMENTAL**
    """
    cmd_dict: T_JSON_DICT = {'method': 'Profiler.startTypeProfile'}
    json = (yield cmd_dict)