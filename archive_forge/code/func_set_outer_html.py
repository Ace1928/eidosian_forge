from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def set_outer_html(node_id: NodeId, outer_html: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets node HTML markup, returns new node id.

    :param node_id: Id of the node to set markup for.
    :param outer_html: Outer HTML markup to set.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['outerHTML'] = outer_html
    cmd_dict: T_JSON_DICT = {'method': 'DOM.setOuterHTML', 'params': params}
    json = (yield cmd_dict)