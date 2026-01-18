from __future__ import annotations
import typing
import warnings
from urwid.split_repr import remove_defaults
from .columns import Columns
from .constants import Align, Sizing, WHSettings
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin
from .divider import Divider
from .monitored_list import MonitoredFocusList, MonitoredList
from .padding import Padding
from .pile import Pile
from .widget import Widget, WidgetError, WidgetWarning, WidgetWrap
@focus_cell.setter
def focus_cell(self, cell: Widget) -> None:
    warnings.warn('only for backwards compatibility.You may also use the new standard container property`focus` to get the focus and `focus_position` to get/set the cell in focus by index', PendingDeprecationWarning, stacklevel=2)
    for i, (w, _options) in enumerate(self.contents):
        if cell == w:
            self.focus_position = i
            return
    raise ValueError(f'Widget not found in GridFlow contents: {cell!r}')