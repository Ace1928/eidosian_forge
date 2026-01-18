from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def set_node_value(node_id: NodeId, value: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets node value for a node with given id.

    :param node_id: Id of the node to set value for.
    :param value: New node's value.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['value'] = value
    cmd_dict: T_JSON_DICT = {'method': 'DOM.setNodeValue', 'params': params}
    json = (yield cmd_dict)