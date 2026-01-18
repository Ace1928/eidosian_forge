from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
def resolve_animation(animation_id: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, runtime.RemoteObject]:
    """
    Gets the remote object of the Animation.

    :param animation_id: Animation id.
    :returns: Corresponding remote object.
    """
    params: T_JSON_DICT = dict()
    params['animationId'] = animation_id
    cmd_dict: T_JSON_DICT = {'method': 'Animation.resolveAnimation', 'params': params}
    json = (yield cmd_dict)
    return runtime.RemoteObject.from_json(json['remoteObject'])