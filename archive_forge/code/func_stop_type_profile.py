from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
def stop_type_profile() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Disable type profile. Disabling releases type profile data collected so far.

    **EXPERIMENTAL**
    """
    cmd_dict: T_JSON_DICT = {'method': 'Profiler.stopTypeProfile'}
    json = (yield cmd_dict)