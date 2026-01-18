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
@focus_position.setter
def focus_position(self, position: int) -> None:
    """
        Set the widget in focus.

        position -- index of child widget to be made focus
        """
    try:
        if position < 0 or position >= len(self.contents):
            raise IndexError(f'No Pile child widget at position {position}')
    except TypeError as exc:
        raise IndexError(f'No Pile child widget at position {position}').with_traceback(exc.__traceback__) from exc
    self.contents.focus = position