from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def take_computed_style_updates() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[dom.NodeId]]:
    """
    Polls the next batch of computed style updates.

    **EXPERIMENTAL**

    :returns: The list of node Ids that have their tracked computed styles updated.
    """
    cmd_dict: T_JSON_DICT = {'method': 'CSS.takeComputedStyleUpdates'}
    json = (yield cmd_dict)
    return [dom.NodeId.from_json(i) for i in json['nodeIds']]