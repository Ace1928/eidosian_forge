from __future__ import annotations
import logging # isort:skip
import itertools
import re
from dataclasses import dataclass
from typing import (
from ..models import (
from ..models.tools import (
from ..util.warnings import warn
def process_tools_arg(plot: Plot, tools: str | Sequence[Tool | str], tooltips: str | tuple[str, str] | None=None) -> tuple[list[Tool], dict[str, Tool]]:
    """ Adds tools to the plot object

    Args:
        plot (Plot): instance of a plot object

        tools (seq[Tool or str]|str): list of tool types or string listing the
            tool names. Those are converted using the to actual Tool instances.

        tooltips (string or seq[tuple[str, str]], optional):
            tooltips to use to configure a HoverTool

    Returns:
        list of Tools objects added to plot, map of supplied string names to tools
    """
    tool_objs, tool_map = _resolve_tools(tools)
    repeated_tools = [str(obj) for obj in _collect_repeated_tools(tool_objs)]
    if repeated_tools:
        warn(f'{','.join(repeated_tools)} are being repeated')
    if tooltips is not None:
        for tool_obj in tool_objs:
            if isinstance(tool_obj, HoverTool):
                tool_obj.tooltips = tooltips
                break
        else:
            tool_objs.append(HoverTool(tooltips=tooltips))
    return (tool_objs, tool_map)