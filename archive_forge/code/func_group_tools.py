from __future__ import annotations
import logging # isort:skip
import math
from collections import defaultdict
from typing import (
from .core.enums import Location, LocationType, SizingModeType
from .core.property.singletons import Undefined, UndefinedType
from .models import (
from .util.dataclasses import dataclass
from .util.warnings import warn
def group_tools(tools: list[Tool | ToolProxy], *, merge: MergeFn[Tool] | None=None, ignore: set[str] | None=None) -> list[Tool | ToolProxy]:
    """ Group common tools into tool proxies. """

    @dataclass
    class ToolEntry:
        tool: Tool
        props: Any
    by_type: defaultdict[type[Tool], list[ToolEntry]] = defaultdict(list)
    computed: list[Tool | ToolProxy] = []
    if ignore is None:
        ignore = {'overlay', 'renderers'}
    for tool in tools:
        if isinstance(tool, ToolProxy):
            computed.append(tool)
        else:
            props = tool.properties_with_values()
            for attr in ignore:
                if attr in props:
                    del props[attr]
            by_type[tool.__class__].append(ToolEntry(tool, props))
    for cls, entries in by_type.items():
        if merge is not None:
            merged = merge(cls, [entry.tool for entry in entries])
            if merged is not None:
                computed.append(merged)
                continue
        while entries:
            head, *tail = entries
            group: list[Tool] = [head.tool]
            for item in list(tail):
                if item.props == head.props:
                    group.append(item.tool)
                    entries.remove(item)
            entries.remove(head)
            if len(group) == 1:
                computed.append(group[0])
            elif merge is not None and (tool := merge(cls, group)) is not None:
                computed.append(tool)
            else:
                computed.append(ToolProxy(tools=group))
    return computed