from __future__ import annotations
import logging # isort:skip
import itertools
import re
from dataclasses import dataclass
from typing import (
from ..models import (
from ..models.tools import (
from ..util.warnings import warn
def process_active_tools(toolbar: Toolbar, tool_map: dict[str, Tool], active_drag: ActiveDrag, active_inspect: ActiveInspect, active_scroll: ActiveScroll, active_tap: ActiveTap, active_multi: ActiveMulti) -> None:
    """ Adds tools to the plot object

    Args:
        toolbar (Toolbar): instance of a Toolbar object
        tools_map (dict[str]): tool_map from _process_tools_arg
        active_drag (str, None, "auto" or Tool): the tool to set active for drag
        active_inspect (str, None, "auto", Tool or Tool[]): the tool to set active for inspect
        active_scroll (str, None, "auto" or Tool): the tool to set active for scroll
        active_tap (str, None, "auto" or Tool): the tool to set active for tap
        active_multi (str, None, "auto" or Tool): the tool to set active for tap

    Returns:
        None

    Note:
        This function sets properties on Toolbar
    """
    if active_drag in ['auto', None] or isinstance(active_drag, Tool):
        toolbar.active_drag = cast(Any, active_drag)
    elif active_drag in tool_map:
        toolbar.active_drag = cast(Any, tool_map[active_drag])
    else:
        raise ValueError(f"Got unknown {active_drag!r} for 'active_drag', which was not a string supplied in 'tools' argument")
    if active_inspect in ['auto', None] or isinstance(active_inspect, Tool) or (isinstance(active_inspect, list) and all((isinstance(t, Tool) for t in active_inspect))):
        toolbar.active_inspect = cast(Any, active_inspect)
    elif isinstance(active_inspect, str) and active_inspect in tool_map:
        toolbar.active_inspect = cast(Any, tool_map[active_inspect])
    else:
        raise ValueError(f"Got unknown {active_inspect!r} for 'active_inspect', which was not a string supplied in 'tools' argument")
    if active_scroll in ['auto', None] or isinstance(active_scroll, Tool):
        toolbar.active_scroll = cast(Any, active_scroll)
    elif active_scroll in tool_map:
        toolbar.active_scroll = cast(Any, tool_map[active_scroll])
    else:
        raise ValueError(f"Got unknown {active_scroll!r} for 'active_scroll', which was not a string supplied in 'tools' argument")
    if active_tap in ['auto', None] or isinstance(active_tap, Tool):
        toolbar.active_tap = cast(Any, active_tap)
    elif active_tap in tool_map:
        toolbar.active_tap = cast(Any, tool_map[active_tap])
    else:
        raise ValueError(f"Got unknown {active_tap!r} for 'active_tap', which was not a string supplied in 'tools' argument")
    if active_multi in ['auto', None] or isinstance(active_multi, Tool):
        toolbar.active_multi = cast(Any, active_multi)
    elif active_multi in tool_map:
        toolbar.active_multi = cast(Any, tool_map[active_multi])
    else:
        raise ValueError(f"Got unknown {active_multi!r} for 'active_multi', which was not a string supplied in 'tools' argument")