from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def get_top_layer_elements() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[NodeId]]:
    """
    Returns NodeIds of current top layer elements.
    Top layer is rendered closest to the user within a viewport, therefore its elements always
    appear on top of all other content.

    **EXPERIMENTAL**

    :returns: NodeIds of top layer elements
    """
    cmd_dict: T_JSON_DICT = {'method': 'DOM.getTopLayerElements'}
    json = (yield cmd_dict)
    return [NodeId.from_json(i) for i in json['nodeIds']]