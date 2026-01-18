from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def set_show_fps_counter(show: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Requests that backend shows the FPS counter

    :param show: True for showing the FPS counter
    """
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.setShowFPSCounter', 'params': params}
    json = (yield cmd_dict)