from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def dispatch_drag_event(type_: str, x: float, y: float, data: DragData, modifiers: typing.Optional[int]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Dispatches a drag event into the page.

    **EXPERIMENTAL**

    :param type_: Type of the drag event.
    :param x: X coordinate of the event relative to the main frame's viewport in CSS pixels.
    :param y: Y coordinate of the event relative to the main frame's viewport in CSS pixels. 0 refers to the top of the viewport and Y increases as it proceeds towards the bottom of the viewport.
    :param data:
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    """
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['x'] = x
    params['y'] = y
    params['data'] = data.to_json()
    if modifiers is not None:
        params['modifiers'] = modifiers
    cmd_dict: T_JSON_DICT = {'method': 'Input.dispatchDragEvent', 'params': params}
    json = (yield cmd_dict)