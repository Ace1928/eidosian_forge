from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def continue_to_location(location: Location, target_call_frames: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Continues execution until specific location is reached.

    :param location: Location to continue to.
    :param target_call_frames: *(Optional)*
    """
    params: T_JSON_DICT = dict()
    params['location'] = location.to_json()
    if target_call_frames is not None:
        params['targetCallFrames'] = target_call_frames
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.continueToLocation', 'params': params}
    json = (yield cmd_dict)