from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
def stop_precise_coverage() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Disable precise code coverage. Disabling releases unnecessary execution count records and allows
    executing optimized code.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Profiler.stopPreciseCoverage'}
    json = (yield cmd_dict)