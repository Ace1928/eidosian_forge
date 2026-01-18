from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def set_show_scroll_bottleneck_rects(show: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Requests that backend shows scroll bottleneck rects

    :param show: True for showing scroll bottleneck rects
    """
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.setShowScrollBottleneckRects', 'params': params}
    json = (yield cmd_dict)