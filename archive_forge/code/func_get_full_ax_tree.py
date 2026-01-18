from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
def get_full_ax_tree() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[AXNode]]:
    """
    Fetches the entire accessibility tree

    **EXPERIMENTAL**

    :returns: 
    """
    cmd_dict: T_JSON_DICT = {'method': 'Accessibility.getFullAXTree'}
    json = (yield cmd_dict)
    return [AXNode.from_json(i) for i in json['nodes']]