from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
def get_window_for_target(target_id: typing.Optional[target.TargetID]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[WindowID, Bounds]]:
    """
    Get the browser window that contains the devtools target.

    **EXPERIMENTAL**

    :param target_id: *(Optional)* Devtools agent host id. If called as a part of the session, associated targetId is used.
    :returns: A tuple with the following items:

        0. **windowId** - Browser window id.
        1. **bounds** - Bounds information of the window. When window state is 'minimized', the restored window position and size are returned.
    """
    params: T_JSON_DICT = dict()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Browser.getWindowForTarget', 'params': params}
    json = (yield cmd_dict)
    return (WindowID.from_json(json['windowId']), Bounds.from_json(json['bounds']))