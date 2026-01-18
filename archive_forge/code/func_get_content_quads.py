from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def get_content_quads(node_id: typing.Optional[NodeId]=None, backend_node_id: typing.Optional[BackendNodeId]=None, object_id: typing.Optional[runtime.RemoteObjectId]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[Quad]]:
    """
    Returns quads that describe node position on the page. This method
    might return multiple quads for inline nodes.

    **EXPERIMENTAL**

    :param node_id: *(Optional)* Identifier of the node.
    :param backend_node_id: *(Optional)* Identifier of the backend node.
    :param object_id: *(Optional)* JavaScript object id of the node wrapper.
    :returns: Quads that describe node layout relative to viewport.
    """
    params: T_JSON_DICT = dict()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'DOM.getContentQuads', 'params': params}
    json = (yield cmd_dict)
    return [Quad.from_json(i) for i in json['quads']]