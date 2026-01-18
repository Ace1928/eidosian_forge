from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def force_pseudo_state(node_id: dom.NodeId, forced_pseudo_classes: typing.List[str]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Ensures that the given node will have specified pseudo-classes whenever its style is computed by
    the browser.

    :param node_id: The element id for which to force the pseudo state.
    :param forced_pseudo_classes: Element pseudo classes to force when computing the element's style.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['forcedPseudoClasses'] = [i for i in forced_pseudo_classes]
    cmd_dict: T_JSON_DICT = {'method': 'CSS.forcePseudoState', 'params': params}
    json = (yield cmd_dict)