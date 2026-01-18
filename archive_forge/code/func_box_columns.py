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
@box_columns.setter
def box_columns(self, box_columns):
    warnings.warn('only for backwards compatibility.You should use the new standard container property `contents`', PendingDeprecationWarning, stacklevel=2)
    box_columns = set(box_columns)
    self.contents = [(w, (t, n, i in box_columns)) for i, (w, (t, n, b)) in enumerate(self.contents)]