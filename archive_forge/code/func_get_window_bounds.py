from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
def get_window_bounds(window_id: WindowID) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, Bounds]:
    """
    Get position and size of the browser window.

    **EXPERIMENTAL**

    :param window_id: Browser window id.
    :returns: Bounds information of the window. When window state is 'minimized', the restored window position and size are returned.
    """
    params: T_JSON_DICT = dict()
    params['windowId'] = window_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Browser.getWindowBounds', 'params': params}
    json = (yield cmd_dict)
    return Bounds.from_json(json['bounds'])