from __future__ import annotations
import typing
import warnings
from itertools import chain, repeat
import urwid
from urwid.canvas import Canvas, CanvasJoin, CompositeCanvas, SolidCanvas
from urwid.command_map import Command
from urwid.split_repr import remove_defaults
from urwid.util import is_mouse_press
from .constants import Align, Sizing, WHSettings
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin, _ContainerElementSizingFlag
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, WidgetError, WidgetWarning
@column_types.setter
def column_types(self, column_types):
    warnings.warn('for backwards compatibility only.You should use the new standard container property .contents to modify Pile contents.', PendingDeprecationWarning, stacklevel=2)
    focus_position = self.focus_position
    self.contents = [(w, ({Sizing.FIXED: WHSettings.GIVEN, Sizing.FLOW: WHSettings.PACK}.get(new_t, new_t), new_n, b)) for (new_t, new_n), (w, (t, n, b)) in zip(column_types, self.contents)]
    if focus_position < len(column_types):
        self.focus_position = focus_position