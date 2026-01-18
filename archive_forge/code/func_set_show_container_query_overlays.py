from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def set_show_container_query_overlays(container_query_highlight_configs: typing.List[ContainerQueryHighlightConfig]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    :param container_query_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    """
    params: T_JSON_DICT = dict()
    params['containerQueryHighlightConfigs'] = [i.to_json() for i in container_query_highlight_configs]
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.setShowContainerQueryOverlays', 'params': params}
    json = (yield cmd_dict)