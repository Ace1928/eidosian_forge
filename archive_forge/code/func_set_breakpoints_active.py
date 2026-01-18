from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def set_breakpoints_active(active: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Activates / deactivates all breakpoints on the page.

    :param active: New value for breakpoints active state.
    """
    params: T_JSON_DICT = dict()
    params['active'] = active
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.setBreakpointsActive', 'params': params}
    json = (yield cmd_dict)