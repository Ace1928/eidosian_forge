from __future__ import annotations
import typing
import warnings
from .attr_map import AttrMap
def get_focus_attr(self) -> Hashable | None:
    focus_map = self.focus_map
    if focus_map:
        return focus_map[None]
    return None