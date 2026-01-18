from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def synthesize_pinch_gesture(x: float, y: float, scale_factor: float, relative_speed: typing.Optional[int]=None, gesture_source_type: typing.Optional[GestureSourceType]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Synthesizes a pinch gesture over a time period by issuing appropriate touch events.

    **EXPERIMENTAL**

    :param x: X coordinate of the start of the gesture in CSS pixels.
    :param y: Y coordinate of the start of the gesture in CSS pixels.
    :param scale_factor: Relative scale factor after zooming (>1.0 zooms in, <1.0 zooms out).
    :param relative_speed: *(Optional)* Relative pointer speed in pixels per second (default: 800).
    :param gesture_source_type: *(Optional)* Which type of input events to be generated (default: 'default', which queries the platform for the preferred input type).
    """
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    params['scaleFactor'] = scale_factor
    if relative_speed is not None:
        params['relativeSpeed'] = relative_speed
    if gesture_source_type is not None:
        params['gestureSourceType'] = gesture_source_type.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Input.synthesizePinchGesture', 'params': params}
    json = (yield cmd_dict)