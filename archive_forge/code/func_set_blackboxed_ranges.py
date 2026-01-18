from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def set_blackboxed_ranges(script_id: runtime.ScriptId, positions: typing.List[ScriptPosition]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Makes backend skip steps in the script in blackboxed ranges. VM will try leave blacklisted
    scripts by performing 'step in' several times, finally resorting to 'step out' if unsuccessful.
    Positions array contains positions where blackbox state is changed. First interval isn't
    blackboxed. Array should be sorted.

    **EXPERIMENTAL**

    :param script_id: Id of the script.
    :param positions:
    """
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    params['positions'] = [i.to_json() for i in positions]
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.setBlackboxedRanges', 'params': params}
    json = (yield cmd_dict)