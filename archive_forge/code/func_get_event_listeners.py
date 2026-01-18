from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
def get_event_listeners(object_id: runtime.RemoteObjectId, depth: typing.Optional[int]=None, pierce: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[EventListener]]:
    """
    Returns event listeners of the given object.

    :param object_id: Identifier of the object to return listeners for.
    :param depth: *(Optional)* The maximum depth at which Node children should be retrieved, defaults to 1. Use -1 for the entire subtree or provide an integer larger than 0.
    :param pierce: *(Optional)* Whether or not iframes and shadow roots should be traversed when returning the subtree (default is false). Reports listeners for all contexts if pierce is enabled.
    :returns: Array of relevant listeners.
    """
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    if depth is not None:
        params['depth'] = depth
    if pierce is not None:
        params['pierce'] = pierce
    cmd_dict: T_JSON_DICT = {'method': 'DOMDebugger.getEventListeners', 'params': params}
    json = (yield cmd_dict)
    return [EventListener.from_json(i) for i in json['listeners']]