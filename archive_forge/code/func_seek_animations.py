from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
def seek_animations(animations: typing.List[str], current_time: float) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Seek a set of animations to a particular time within each animation.

    :param animations: List of animation ids to seek.
    :param current_time: Set the current time of each animation.
    """
    params: T_JSON_DICT = dict()
    params['animations'] = [i for i in animations]
    params['currentTime'] = current_time
    cmd_dict: T_JSON_DICT = {'method': 'Animation.seekAnimations', 'params': params}
    json = (yield cmd_dict)