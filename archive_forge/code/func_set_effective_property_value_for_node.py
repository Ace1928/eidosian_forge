from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def set_effective_property_value_for_node(node_id: dom.NodeId, property_name: str, value: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Find a rule with the given active property for the given node and set the new value for this
    property

    :param node_id: The element id for which to set property.
    :param property_name:
    :param value:
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['propertyName'] = property_name
    params['value'] = value
    cmd_dict: T_JSON_DICT = {'method': 'CSS.setEffectivePropertyValueForNode', 'params': params}
    json = (yield cmd_dict)