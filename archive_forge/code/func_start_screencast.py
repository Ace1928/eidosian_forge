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
def start_screencast(format_: typing.Optional[str]=None, quality: typing.Optional[int]=None, max_width: typing.Optional[int]=None, max_height: typing.Optional[int]=None, every_nth_frame: typing.Optional[int]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Starts sending each frame using the ``screencastFrame`` event.

    **EXPERIMENTAL**

    :param format_: *(Optional)* Image compression format.
    :param quality: *(Optional)* Compression quality from range [0..100].
    :param max_width: *(Optional)* Maximum screenshot width.
    :param max_height: *(Optional)* Maximum screenshot height.
    :param every_nth_frame: *(Optional)* Send every n-th frame.
    """
    params: T_JSON_DICT = dict()
    if format_ is not None:
        params['format'] = format_
    if quality is not None:
        params['quality'] = quality
    if max_width is not None:
        params['maxWidth'] = max_width
    if max_height is not None:
        params['maxHeight'] = max_height
    if every_nth_frame is not None:
        params['everyNthFrame'] = every_nth_frame
    cmd_dict: T_JSON_DICT = {'method': 'Page.startScreencast', 'params': params}
    json = (yield cmd_dict)