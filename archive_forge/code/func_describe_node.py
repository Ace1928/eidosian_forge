from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def describe_node(node_id: typing.Optional[NodeId]=None, backend_node_id: typing.Optional[BackendNodeId]=None, object_id: typing.Optional[runtime.RemoteObjectId]=None, depth: typing.Optional[int]=None, pierce: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, Node]:
    """
    Describes node given its id, does not require domain to be enabled. Does not start tracking any
    objects, can be used for automation.

    :param node_id: *(Optional)* Identifier of the node.
    :param backend_node_id: *(Optional)* Identifier of the backend node.
    :param object_id: *(Optional)* JavaScript object id of the node wrapper.
    :param depth: *(Optional)* The maximum depth at which children should be retrieved, defaults to 1. Use -1 for the entire subtree or provide an integer larger than 0.
    :param pierce: *(Optional)* Whether or not iframes and shadow roots should be traversed when returning the subtree (default is false).
    :returns: Node description.
    """
    params: T_JSON_DICT = dict()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    if depth is not None:
        params['depth'] = depth
    if pierce is not None:
        params['pierce'] = pierce
    cmd_dict: T_JSON_DICT = {'method': 'DOM.describeNode', 'params': params}
    json = (yield cmd_dict)
    return Node.from_json(json['node'])