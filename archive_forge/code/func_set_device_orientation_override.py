from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def set_device_orientation_override(alpha: float, beta: float, gamma: float) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Overrides the Device Orientation.

    **EXPERIMENTAL**

    :param alpha: Mock alpha
    :param beta: Mock beta
    :param gamma: Mock gamma
    """
    params: T_JSON_DICT = dict()
    params['alpha'] = alpha
    params['beta'] = beta
    params['gamma'] = gamma
    cmd_dict: T_JSON_DICT = {'method': 'Page.setDeviceOrientationOverride', 'params': params}
    json = (yield cmd_dict)