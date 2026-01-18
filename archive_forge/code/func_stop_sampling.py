from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def stop_sampling() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, SamplingHeapProfile]:
    """


    :returns: Recorded sampling heap profile.
    """
    cmd_dict: T_JSON_DICT = {'method': 'HeapProfiler.stopSampling'}
    json = (yield cmd_dict)
    return SamplingHeapProfile.from_json(json['profile'])