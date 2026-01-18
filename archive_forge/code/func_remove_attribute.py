from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def remove_attribute(node_id: NodeId, name: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Removes attribute with given name from an element with given id.

    :param node_id: Id of the element to remove attribute from.
    :param name: Name of the attribute to remove.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['name'] = name
    cmd_dict: T_JSON_DICT = {'method': 'DOM.removeAttribute', 'params': params}
    json = (yield cmd_dict)