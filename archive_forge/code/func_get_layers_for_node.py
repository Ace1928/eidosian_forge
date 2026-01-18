from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def get_layers_for_node(node_id: dom.NodeId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, CSSLayerData]:
    """
    Returns all layers parsed by the rendering engine for the tree scope of a node.
    Given a DOM element identified by nodeId, getLayersForNode returns the root
    layer for the nearest ancestor document or shadow root. The layer root contains
    the full layer tree for the tree scope and their ordering.

    **EXPERIMENTAL**

    :param node_id:
    :returns: 
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'CSS.getLayersForNode', 'params': params}
    json = (yield cmd_dict)
    return CSSLayerData.from_json(json['rootLayer'])