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
@focus_col.setter
def focus_col(self, new_position) -> None:
    warnings.warn('only for backwards compatibility.You may also use the new standard container property `focus_position` to get the focus.', PendingDeprecationWarning, stacklevel=2)
    self.focus_position = new_position