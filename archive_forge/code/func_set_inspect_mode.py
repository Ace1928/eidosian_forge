from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def set_inspect_mode(mode: InspectMode, highlight_config: typing.Optional[HighlightConfig]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enters the 'inspect' mode. In this mode, elements that user is hovering over are highlighted.
    Backend then generates 'inspectNodeRequested' event upon element selection.

    :param mode: Set an inspection mode.
    :param highlight_config: *(Optional)* A descriptor for the highlight appearance of hovered-over nodes. May be omitted if ```enabled == false```.
    """
    params: T_JSON_DICT = dict()
    params['mode'] = mode.to_json()
    if highlight_config is not None:
        params['highlightConfig'] = highlight_config.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.setInspectMode', 'params': params}
    json = (yield cmd_dict)