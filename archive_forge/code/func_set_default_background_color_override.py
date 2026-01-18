from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_default_background_color_override(color: typing.Optional[dom.RGBA]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets or clears an override of the default background color of the frame. This override is used
    if the content does not specify one.

    :param color: *(Optional)* RGBA of the default background color. If not specified, any existing override will be cleared.
    """
    params: T_JSON_DICT = dict()
    if color is not None:
        params['color'] = color.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setDefaultBackgroundColorOverride', 'params': params}
    json = (yield cmd_dict)