from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_automation_override(enabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Allows overriding the automation flag.

    **EXPERIMENTAL**

    :param enabled: Whether the override should be enabled.
    """
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setAutomationOverride', 'params': params}
    json = (yield cmd_dict)