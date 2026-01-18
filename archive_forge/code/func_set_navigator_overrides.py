from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_navigator_overrides(platform: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Overrides value returned by the javascript navigator object.

    **EXPERIMENTAL**

    :param platform: The platform navigator.platform should return.
    """
    params: T_JSON_DICT = dict()
    params['platform'] = platform
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setNavigatorOverrides', 'params': params}
    json = (yield cmd_dict)