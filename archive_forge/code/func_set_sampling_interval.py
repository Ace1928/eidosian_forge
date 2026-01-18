from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
def set_sampling_interval(interval: int) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Changes CPU profiler sampling interval. Must be called before CPU profiles recording started.

    :param interval: New sampling interval in microseconds.
    """
    params: T_JSON_DICT = dict()
    params['interval'] = interval
    cmd_dict: T_JSON_DICT = {'method': 'Profiler.setSamplingInterval', 'params': params}
    json = (yield cmd_dict)