from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def get_frame_owner(frame_id: page.FrameId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[BackendNodeId, typing.Optional[NodeId]]]:
    """
    Returns iframe node that owns iframe with the given domain.

    **EXPERIMENTAL**

    :param frame_id:
    :returns: A tuple with the following items:

        0. **backendNodeId** - Resulting node.
        1. **nodeId** - *(Optional)* Id of the node at given coordinates, only when enabled and requested document.
    """
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'DOM.getFrameOwner', 'params': params}
    json = (yield cmd_dict)
    return (BackendNodeId.from_json(json['backendNodeId']), NodeId.from_json(json['nodeId']) if 'nodeId' in json else None)