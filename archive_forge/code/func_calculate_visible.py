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
def calculate_visible(self, size: tuple[int, int], focus: bool=False) -> VisibleInfo | tuple[None, None, None]:
    """
        Returns the widgets that would be displayed in
        the ListBox given the current *size* and *focus*.

        see :meth:`Widget.render` for parameter details

        :returns: (*middle*, *top*, *bottom*) or (``None``, ``None``, ``None``)

        *middle*
            (*row offset*(when +ve) or *inset*(when -ve),
            *focus widget*, *focus position*, *focus rows*,
            *cursor coords* or ``None``)
        *top*
            (*# lines to trim off top*,
            list of (*widget*, *position*, *rows*) tuples above focus in order from bottom to top)
        *bottom*
            (*# lines to trim off bottom*,
            list of (*widget*, *position*, *rows*) tuples below focus in order from top to bottom)
        """
    maxcol, maxrow = size
    if self.set_focus_pending or self.set_focus_valign_pending:
        self._set_focus_complete((maxcol, maxrow), focus)
    focus_widget, focus_pos = self._body.get_focus()
    if focus_widget is None:
        return (None, None, None)
    top_pos = focus_pos
    offset_rows, inset_rows = self.get_focus_offset_inset((maxcol, maxrow))
    if maxrow and offset_rows >= maxrow:
        offset_rows = maxrow - 1
    cursor = None
    if maxrow and focus_widget.selectable() and focus and hasattr(focus_widget, 'get_cursor_coords'):
        cursor = focus_widget.get_cursor_coords((maxcol,))
    if cursor is not None:
        _cx, cy = cursor
        effective_cy = cy + offset_rows - inset_rows
        if effective_cy < 0:
            inset_rows = cy
        elif effective_cy >= maxrow:
            offset_rows = maxrow - cy - 1
            if offset_rows < 0:
                inset_rows, offset_rows = (-offset_rows, 0)
    trim_top = inset_rows
    focus_rows = focus_widget.rows((maxcol,), True)
    pos = focus_pos
    fill_lines = offset_rows
    fill_above = []
    top_pos = pos
    while fill_lines > 0:
        prev, pos = self._body.get_prev(pos)
        if prev is None:
            offset_rows -= fill_lines
            break
        top_pos = pos
        p_rows = prev.rows((maxcol,))
        if p_rows:
            fill_above.append(VisibleInfoFillItem(prev, pos, p_rows))
        if p_rows > fill_lines:
            trim_top = p_rows - fill_lines
            break
        fill_lines -= p_rows
    trim_bottom = max(focus_rows + offset_rows - inset_rows - maxrow, 0)
    pos = focus_pos
    fill_lines = maxrow - focus_rows - offset_rows + inset_rows
    fill_below = []
    while fill_lines > 0:
        next_pos, pos = self._body.get_next(pos)
        if next_pos is None:
            break
        n_rows = next_pos.rows((maxcol,))
        if n_rows:
            fill_below.append(VisibleInfoFillItem(next_pos, pos, n_rows))
        if n_rows > fill_lines:
            trim_bottom = n_rows - fill_lines
            fill_lines -= n_rows
            break
        fill_lines -= n_rows
    fill_lines = max(0, fill_lines)
    if fill_lines > 0 and trim_top > 0:
        if fill_lines <= trim_top:
            trim_top -= fill_lines
            offset_rows += fill_lines
            fill_lines = 0
        else:
            fill_lines -= trim_top
            offset_rows += trim_top
            trim_top = 0
    pos = top_pos
    while fill_lines > 0:
        prev, pos = self._body.get_prev(pos)
        if prev is None:
            break
        p_rows = prev.rows((maxcol,))
        fill_above.append(VisibleInfoFillItem(prev, pos, p_rows))
        if p_rows > fill_lines:
            trim_top = p_rows - fill_lines
            offset_rows += fill_lines
            break
        fill_lines -= p_rows
        offset_rows += p_rows
    return VisibleInfo(VisibleInfoMiddle(offset_rows - inset_rows, focus_widget, focus_pos, focus_rows, cursor), VisibleInfoTopBottom(trim_top, fill_above), VisibleInfoTopBottom(trim_bottom, fill_below))