from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def get_child_ax_nodes(id_: AXNodeId, frame_id: typing.Optional[page.FrameId]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[AXNode]]:
    """
    Fetches a particular accessibility node by AXNodeId.
    Requires ``enable()`` to have been called previously.

    **EXPERIMENTAL**

    :param id_:
    :param frame_id: *(Optional)* The frame in whose document the node resides. If omitted, the root frame is used.
    :returns: 
    """
    params: T_JSON_DICT = dict()
    params['id'] = id_.to_json()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Accessibility.getChildAXNodes', 'params': params}
    json = (yield cmd_dict)
    return [AXNode.from_json(i) for i in json['nodes']]