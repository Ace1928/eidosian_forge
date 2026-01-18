from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
def remove_dom_breakpoint(node_id: dom.NodeId, type_: DOMBreakpointType) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Removes DOM breakpoint that was set using ``setDOMBreakpoint``.

    :param node_id: Identifier of the node to remove breakpoint from.
    :param type_: Type of the breakpoint to remove.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['type'] = type_.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'DOMDebugger.removeDOMBreakpoint', 'params': params}
    json = (yield cmd_dict)