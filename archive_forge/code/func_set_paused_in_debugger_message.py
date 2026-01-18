from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def set_paused_in_debugger_message(message: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    :param message: *(Optional)* The message to display, also triggers resume and step over controls.
    """
    params: T_JSON_DICT = dict()
    if message is not None:
        params['message'] = message
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.setPausedInDebuggerMessage', 'params': params}
    json = (yield cmd_dict)