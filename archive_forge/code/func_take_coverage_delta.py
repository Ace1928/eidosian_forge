from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def take_coverage_delta() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[typing.List[RuleUsage], float]]:
    """
    Obtain list of rules that became used since last call to this method (or since start of coverage
    instrumentation).

    :returns: A tuple with the following items:

        0. **coverage** - 
        1. **timestamp** - Monotonically increasing time, in seconds.
    """
    cmd_dict: T_JSON_DICT = {'method': 'CSS.takeCoverageDelta'}
    json = (yield cmd_dict)
    return ([RuleUsage.from_json(i) for i in json['coverage']], float(json['timestamp']))