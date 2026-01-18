from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def get_outer_html(node_id: typing.Optional[NodeId]=None, backend_node_id: typing.Optional[BackendNodeId]=None, object_id: typing.Optional[runtime.RemoteObjectId]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, str]:
    """
    Returns node's HTML markup.

    :param node_id: *(Optional)* Identifier of the node.
    :param backend_node_id: *(Optional)* Identifier of the backend node.
    :param object_id: *(Optional)* JavaScript object id of the node wrapper.
    :returns: Outer HTML markup.
    """
    params: T_JSON_DICT = dict()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'DOM.getOuterHTML', 'params': params}
    json = (yield cmd_dict)
    return str(json['outerHTML'])