from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def set_show_isolated_elements(isolated_element_highlight_configs: typing.List[IsolatedElementHighlightConfig]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Show elements in isolation mode with overlays.

    :param isolated_element_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    """
    params: T_JSON_DICT = dict()
    params['isolatedElementHighlightConfigs'] = [i.to_json() for i in isolated_element_highlight_configs]
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.setShowIsolatedElements', 'params': params}
    json = (yield cmd_dict)