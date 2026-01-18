from __future__ import annotations
import typing
from urwid.canvas import CompositeCanvas
from .constants import Sizing
from .overlay import Overlay
from .widget import delegate_to_widget_mixin
from .widget_decoration import WidgetDecoration
def open_pop_up(self) -> None:
    self._pop_up_widget = self.create_pop_up()
    self._invalidate()