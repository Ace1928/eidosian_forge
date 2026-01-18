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
def get_focus_offset_inset(self, size: tuple[int, int]) -> tuple[int, int]:
    """Return (offset rows, inset rows) for focus widget."""
    maxcol, _maxrow = size
    focus_widget, _pos = self._body.get_focus()
    focus_rows = focus_widget.rows((maxcol,), True)
    offset_rows = self.offset_rows
    inset_rows = 0
    if offset_rows == 0:
        inum, iden = self.inset_fraction
        if inum < 0 or iden < 0 or inum >= iden:
            raise ListBoxError(f'Invalid inset_fraction: {self.inset_fraction!r}')
        inset_rows = focus_rows * inum // iden
        if inset_rows and inset_rows >= focus_rows:
            raise ListBoxError('urwid inset_fraction error (please report)')
    return (offset_rows, inset_rows)