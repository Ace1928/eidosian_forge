from __future__ import annotations
import operator
import typing
import warnings
from collections.abc import Iterable, Sized
from contextlib import suppress
from typing_extensions import Protocol, runtime_checkable
from urwid import signals
from urwid.canvas import CanvasCombine, SolidCanvas
from .constants import Sizing, VAlign, WHSettings, normalize_valign
from .container import WidgetContainerMixin
from .filler import calculate_top_bottom_filler
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, nocache_widget_render_instance
def get_scrollpos(self, size: tuple[int, int] | None=None, focus: bool=False) -> int:
    """Current scrolling position."""
    self._check_support_scrolling()
    if not self._body:
        return 0
    if size is not None:
        self._rendered_size = size
    mid, top, _bottom = self.calculate_visible(self._rendered_size, focus)
    start_row = top.trim
    maxcol = self._rendered_size[0]
    if top.fill:
        pos = top.fill[-1].position
    else:
        pos = mid.focus_pos
    prev, pos = self._body.get_prev(pos)
    while prev is not None:
        start_row += prev.rows((maxcol,))
        prev, pos = self._body.get_prev(pos)
    return start_row