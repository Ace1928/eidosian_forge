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
def get_item_size(self, size: tuple[()] | tuple[int] | tuple[int, int], i: int, focus: bool, item_rows: list[int] | None=None) -> tuple[()] | tuple[int] | tuple[int, int]:
    """
        Return a size appropriate for passing to self.contents[i][0].render
        """
    _w, (f, height) = self.contents[i]
    if f == WHSettings.PACK:
        if not size:
            return ()
        return (size[0],)
    if not size:
        raise PileError(f'Element {i} using parameters {f} and do not have full size information')
    maxcol = size[0]
    if f == WHSettings.GIVEN:
        return (maxcol, height)
    if f == WHSettings.WEIGHT:
        if len(size) == 2:
            if not item_rows:
                item_rows = self.get_item_rows(size, focus)
            return (maxcol, item_rows[i])
        return (maxcol,)
    raise PileError(f'Unsupported item height rules: {f}')