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
def rows_max(self, size: tuple[int, int] | None=None, focus: bool=False) -> int:
    """Scrollable protocol for sized iterable and not wrapped around contents."""
    self._check_support_scrolling()
    if size is not None:
        self._rendered_size = size
    if size or not self._rows_max_cached:
        cols = self._rendered_size[0]
        rows = 0
        focused_w, idx = self.body.get_focus()
        rows += focused_w.rows((cols,), focus)
        prev, pos = self._body.get_prev(idx)
        while prev is not None:
            rows += prev.rows((cols,), False)
            prev, pos = self._body.get_prev(pos)
        next_, pos = self.body.get_next(idx)
        while next_ is not None:
            rows += next_.rows((cols,), True)
            next_, pos = self._body.get_next(pos)
        self._rows_max_cached = rows
    return self._rows_max_cached