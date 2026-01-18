from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def set_show_viewport_size_on_resize(show: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Paints viewport size upon main frame resize.

    :param show: Whether to paint size or not.
    """
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.setShowViewportSizeOnResize', 'params': params}
    json = (yield cmd_dict)