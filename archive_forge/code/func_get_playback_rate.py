from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
def get_playback_rate() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, float]:
    """
    Gets the playback rate of the document timeline.

    :returns: Playback rate for animations on page.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Animation.getPlaybackRate'}
    json = (yield cmd_dict)
    return float(json['playbackRate'])