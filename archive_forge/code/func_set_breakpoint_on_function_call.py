from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def set_breakpoint_on_function_call(object_id: runtime.RemoteObjectId, condition: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, BreakpointId]:
    """
    Sets JavaScript breakpoint before each call to the given function.
    If another function was created from the same source as a given one,
    calling it will also trigger the breakpoint.

    **EXPERIMENTAL**

    :param object_id: Function object id.
    :param condition: *(Optional)* Expression to use as a breakpoint condition. When specified, debugger will stop on the breakpoint if this expression evaluates to true.
    :returns: Id of the created breakpoint for further reference.
    """
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    if condition is not None:
        params['condition'] = condition
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.setBreakpointOnFunctionCall', 'params': params}
    json = (yield cmd_dict)
    return BreakpointId.from_json(json['breakpointId'])