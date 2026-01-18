from __future__ import annotations
import typing
import warnings
from itertools import chain, repeat
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.command_map import Command
from urwid.split_repr import remove_defaults
from urwid.util import is_mouse_press
from .constants import Sizing, WHSettings
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin, _ContainerElementSizingFlag
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, WidgetError, WidgetWarning
def set_focus(self, item: Widget | int) -> None:
    warnings.warn('for backwards compatibility.You may also use the new standard container property .focus to get the child widget in focus.', PendingDeprecationWarning, stacklevel=2)
    if isinstance(item, int):
        self.focus_position = item
        return
    for i, (w, _options) in enumerate(self.contents):
        if item == w:
            self.focus_position = i
            return
    raise ValueError(f'Widget not found in Pile contents: {item!r}')