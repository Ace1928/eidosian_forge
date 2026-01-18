from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def get_source_order_highlight_object_for_test(node_id: dom.NodeId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, dict]:
    """
    For Source Order Viewer testing.

    :param node_id: Id of the node to highlight.
    :returns: Source order highlight data for the node id provided.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.getSourceOrderHighlightObjectForTest', 'params': params}
    json = (yield cmd_dict)
    return dict(json['highlight'])