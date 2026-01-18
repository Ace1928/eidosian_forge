from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def get_nodes_for_subtree_by_style(node_id: NodeId, computed_styles: typing.List[CSSComputedStyleProperty], pierce: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[NodeId]]:
    """
    Finds nodes with a given computed style in a subtree.

    **EXPERIMENTAL**

    :param node_id: Node ID pointing to the root of a subtree.
    :param computed_styles: The style to filter nodes by (includes nodes if any of properties matches).
    :param pierce: *(Optional)* Whether or not iframes and shadow roots in the same target should be traversed when returning the results (default is false).
    :returns: Resulting nodes.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['computedStyles'] = [i.to_json() for i in computed_styles]
    if pierce is not None:
        params['pierce'] = pierce
    cmd_dict: T_JSON_DICT = {'method': 'DOM.getNodesForSubtreeByStyle', 'params': params}
    json = (yield cmd_dict)
    return [NodeId.from_json(i) for i in json['nodeIds']]