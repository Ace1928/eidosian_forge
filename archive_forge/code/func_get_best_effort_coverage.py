from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
def get_best_effort_coverage() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[ScriptCoverage]]:
    """
    Collect coverage data for the current isolate. The coverage data may be incomplete due to
    garbage collection.

    :returns: Coverage data for the current isolate.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Profiler.getBestEffortCoverage'}
    json = (yield cmd_dict)
    return [ScriptCoverage.from_json(i) for i in json['result']]