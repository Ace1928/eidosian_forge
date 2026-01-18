from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def emulate_touch_from_mouse_event(type_: str, x: int, y: int, button: MouseButton, timestamp: typing.Optional[TimeSinceEpoch]=None, delta_x: typing.Optional[float]=None, delta_y: typing.Optional[float]=None, modifiers: typing.Optional[int]=None, click_count: typing.Optional[int]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Emulates touch event from the mouse event parameters.

    **EXPERIMENTAL**

    :param type_: Type of the mouse event.
    :param x: X coordinate of the mouse pointer in DIP.
    :param y: Y coordinate of the mouse pointer in DIP.
    :param button: Mouse button. Only "none", "left", "right" are supported.
    :param timestamp: *(Optional)* Time at which the event occurred (default: current time).
    :param delta_x: *(Optional)* X delta in DIP for mouse wheel event (default: 0).
    :param delta_y: *(Optional)* Y delta in DIP for mouse wheel event (default: 0).
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    :param click_count: *(Optional)* Number of times the mouse button was clicked (default: 0).
    """
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['x'] = x
    params['y'] = y
    params['button'] = button.to_json()
    if timestamp is not None:
        params['timestamp'] = timestamp.to_json()
    if delta_x is not None:
        params['deltaX'] = delta_x
    if delta_y is not None:
        params['deltaY'] = delta_y
    if modifiers is not None:
        params['modifiers'] = modifiers
    if click_count is not None:
        params['clickCount'] = click_count
    cmd_dict: T_JSON_DICT = {'method': 'Input.emulateTouchFromMouseEvent', 'params': params}
    json = (yield cmd_dict)