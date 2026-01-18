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
@item_types.setter
def item_types(self, item_types):
    warnings.warn('only for backwards compatibility. You should use the new standard container property `contents`', PendingDeprecationWarning, stacklevel=2)
    focus_position = self.focus_position
    self.contents = [(w, ({Sizing.FIXED: WHSettings.GIVEN, Sizing.FLOW: WHSettings.PACK}.get(new_t, new_t), new_height)) for (new_t, new_height), (w, options) in zip(item_types, self.contents)]
    if focus_position < len(item_types):
        self.focus_position = focus_position