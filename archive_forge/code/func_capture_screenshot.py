from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def capture_screenshot(format_: typing.Optional[str]=None, quality: typing.Optional[int]=None, clip: typing.Optional[Viewport]=None, from_surface: typing.Optional[bool]=None, capture_beyond_viewport: typing.Optional[bool]=None, optimize_for_speed: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, str]:
    """
    Capture page screenshot.

    :param format_: *(Optional)* Image compression format (defaults to png).
    :param quality: *(Optional)* Compression quality from range [0..100] (jpeg only).
    :param clip: *(Optional)* Capture the screenshot of a given region only.
    :param from_surface: **(EXPERIMENTAL)** *(Optional)* Capture the screenshot from the surface, rather than the view. Defaults to true.
    :param capture_beyond_viewport: **(EXPERIMENTAL)** *(Optional)* Capture the screenshot beyond the viewport. Defaults to false.
    :param optimize_for_speed: **(EXPERIMENTAL)** *(Optional)* Optimize image encoding for speed, not for resulting size (defaults to false)
    :returns: Base64-encoded image data.
    """
    params: T_JSON_DICT = dict()
    if format_ is not None:
        params['format'] = format_
    if quality is not None:
        params['quality'] = quality
    if clip is not None:
        params['clip'] = clip.to_json()
    if from_surface is not None:
        params['fromSurface'] = from_surface
    if capture_beyond_viewport is not None:
        params['captureBeyondViewport'] = capture_beyond_viewport
    if optimize_for_speed is not None:
        params['optimizeForSpeed'] = optimize_for_speed
    cmd_dict: T_JSON_DICT = {'method': 'Page.captureScreenshot', 'params': params}
    json = (yield cmd_dict)
    return str(json['data'])