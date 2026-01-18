from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_focus_emulation_enabled(enabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enables or disables simulating a focused and active page.

    **EXPERIMENTAL**

    :param enabled: Whether to enable to disable focus emulation.
    """
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setFocusEmulationEnabled', 'params': params}
    json = (yield cmd_dict)