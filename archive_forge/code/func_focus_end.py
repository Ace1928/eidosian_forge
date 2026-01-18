from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def focus_end(self, size: tuple[int, int]) -> None:
    """Move focus to far bottom."""
    maxrow, _maxcol = size
    _widget, pos = self.body.get_focus()
    lastwidget = pos.get_root().get_widget().last_child()
    if lastwidget:
        lastnode = lastwidget.get_node()
        self.change_focus(size, lastnode, maxrow - 1)