from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_touch_emulation_enabled(enabled: bool, max_touch_points: typing.Optional[int]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enables touch on platforms which do not support them.

    :param enabled: Whether the touch event emulation should be enabled.
    :param max_touch_points: *(Optional)* Maximum touch points supported. Defaults to one.
    """
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    if max_touch_points is not None:
        params['maxTouchPoints'] = max_touch_points
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setTouchEmulationEnabled', 'params': params}
    json = (yield cmd_dict)