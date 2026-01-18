from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def get_inline_styles_for_node(node_id: dom.NodeId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[typing.Optional[CSSStyle], typing.Optional[CSSStyle]]]:
    """
    Returns the styles defined inline (explicitly in the "style" attribute and implicitly, using DOM
    attributes) for a DOM node identified by ``nodeId``.

    :param node_id:
    :returns: A tuple with the following items:

        0. **inlineStyle** - *(Optional)* Inline style for the specified DOM node.
        1. **attributesStyle** - *(Optional)* Attribute-defined element style (e.g. resulting from "width=20 height=100%").
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'CSS.getInlineStylesForNode', 'params': params}
    json = (yield cmd_dict)
    return (CSSStyle.from_json(json['inlineStyle']) if 'inlineStyle' in json else None, CSSStyle.from_json(json['attributesStyle']) if 'attributesStyle' in json else None)