from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
def get_runtime_call_stats() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[CounterInfo]]:
    """
    Retrieve run time call stats.

    **EXPERIMENTAL**

    :returns: Collected counter information.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Profiler.getRuntimeCallStats'}
    json = (yield cmd_dict)
    return [CounterInfo.from_json(i) for i in json['result']]