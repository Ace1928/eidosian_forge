from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def push_node_by_path_to_frontend(path: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, NodeId]:
    """
    Requests that the node is sent to the caller given its path. // FIXME, use XPath

    **EXPERIMENTAL**

    :param path: Path to node in the proprietary format.
    :returns: Id of the node for given path.
    """
    params: T_JSON_DICT = dict()
    params['path'] = path
    cmd_dict: T_JSON_DICT = {'method': 'DOM.pushNodeByPathToFrontend', 'params': params}
    json = (yield cmd_dict)
    return NodeId.from_json(json['nodeId'])