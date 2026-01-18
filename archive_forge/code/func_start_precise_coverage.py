from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
def start_precise_coverage(call_count: typing.Optional[bool]=None, detailed: typing.Optional[bool]=None, allow_triggered_updates: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, float]:
    """
    Enable precise code coverage. Coverage data for JavaScript executed before enabling precise code
    coverage may be incomplete. Enabling prevents running optimized code and resets execution
    counters.

    :param call_count: *(Optional)* Collect accurate call counts beyond simple 'covered' or 'not covered'.
    :param detailed: *(Optional)* Collect block-based coverage.
    :param allow_triggered_updates: *(Optional)* Allow the backend to send updates on its own initiative
    :returns: Monotonically increasing time (in seconds) when the coverage update was taken in the backend.
    """
    params: T_JSON_DICT = dict()
    if call_count is not None:
        params['callCount'] = call_count
    if detailed is not None:
        params['detailed'] = detailed
    if allow_triggered_updates is not None:
        params['allowTriggeredUpdates'] = allow_triggered_updates
    cmd_dict: T_JSON_DICT = {'method': 'Profiler.startPreciseCoverage', 'params': params}
    json = (yield cmd_dict)
    return float(json['timestamp'])