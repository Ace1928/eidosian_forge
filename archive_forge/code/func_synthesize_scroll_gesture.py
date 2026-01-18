from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def synthesize_scroll_gesture(x: float, y: float, x_distance: typing.Optional[float]=None, y_distance: typing.Optional[float]=None, x_overscroll: typing.Optional[float]=None, y_overscroll: typing.Optional[float]=None, prevent_fling: typing.Optional[bool]=None, speed: typing.Optional[int]=None, gesture_source_type: typing.Optional[GestureSourceType]=None, repeat_count: typing.Optional[int]=None, repeat_delay_ms: typing.Optional[int]=None, interaction_marker_name: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Synthesizes a scroll gesture over a time period by issuing appropriate touch events.

    **EXPERIMENTAL**

    :param x: X coordinate of the start of the gesture in CSS pixels.
    :param y: Y coordinate of the start of the gesture in CSS pixels.
    :param x_distance: *(Optional)* The distance to scroll along the X axis (positive to scroll left).
    :param y_distance: *(Optional)* The distance to scroll along the Y axis (positive to scroll up).
    :param x_overscroll: *(Optional)* The number of additional pixels to scroll back along the X axis, in addition to the given distance.
    :param y_overscroll: *(Optional)* The number of additional pixels to scroll back along the Y axis, in addition to the given distance.
    :param prevent_fling: *(Optional)* Prevent fling (default: true).
    :param speed: *(Optional)* Swipe speed in pixels per second (default: 800).
    :param gesture_source_type: *(Optional)* Which type of input events to be generated (default: 'default', which queries the platform for the preferred input type).
    :param repeat_count: *(Optional)* The number of times to repeat the gesture (default: 0).
    :param repeat_delay_ms: *(Optional)* The number of milliseconds delay between each repeat. (default: 250).
    :param interaction_marker_name: *(Optional)* The name of the interaction markers to generate, if not empty (default: "").
    """
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    if x_distance is not None:
        params['xDistance'] = x_distance
    if y_distance is not None:
        params['yDistance'] = y_distance
    if x_overscroll is not None:
        params['xOverscroll'] = x_overscroll
    if y_overscroll is not None:
        params['yOverscroll'] = y_overscroll
    if prevent_fling is not None:
        params['preventFling'] = prevent_fling
    if speed is not None:
        params['speed'] = speed
    if gesture_source_type is not None:
        params['gestureSourceType'] = gesture_source_type.to_json()
    if repeat_count is not None:
        params['repeatCount'] = repeat_count
    if repeat_delay_ms is not None:
        params['repeatDelayMs'] = repeat_delay_ms
    if interaction_marker_name is not None:
        params['interactionMarkerName'] = interaction_marker_name
    cmd_dict: T_JSON_DICT = {'method': 'Input.synthesizeScrollGesture', 'params': params}
    json = (yield cmd_dict)