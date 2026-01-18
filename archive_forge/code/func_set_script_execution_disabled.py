from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_script_execution_disabled(value: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Switches script execution in the page.

    :param value: Whether script execution should be disabled in the page.
    """
    params: T_JSON_DICT = dict()
    params['value'] = value
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setScriptExecutionDisabled', 'params': params}
    json = (yield cmd_dict)