from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
def remove_event_listener_breakpoint(event_name: str, target_name: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Removes breakpoint on particular DOM event.

    :param event_name: Event name.
    :param target_name: **(EXPERIMENTAL)** *(Optional)* EventTarget interface name.
    """
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    if target_name is not None:
        params['targetName'] = target_name
    cmd_dict: T_JSON_DICT = {'method': 'DOMDebugger.removeEventListenerBreakpoint', 'params': params}
    json = (yield cmd_dict)