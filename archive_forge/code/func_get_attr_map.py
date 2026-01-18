from __future__ import annotations
import typing
from collections.abc import Hashable, Mapping
from urwid.canvas import CompositeCanvas
from .widget import WidgetError, delegate_to_widget_mixin
from .widget_decoration import WidgetDecoration
def get_attr_map(self) -> dict[Hashable | None, Hashable]:
    return dict(self._attr_map)