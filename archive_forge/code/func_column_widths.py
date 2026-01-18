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
def column_widths(self, size: tuple[int] | tuple[int, int], focus: bool=False) -> list[int]:
    """
        Return a list of column widths.

        0 values in the list means hide the corresponding column completely
        """
    maxcol = size[0]
    if maxcol == self._cache_maxcol and (not any((t == WHSettings.PACK for w, (t, n, b) in self.contents))):
        return self._cache_column_widths
    widths = []
    weighted = []
    shared = maxcol + self.dividechars
    for i, (w, (t, width, b)) in enumerate(self.contents):
        if t == WHSettings.GIVEN:
            static_w = width
        elif t == WHSettings.PACK:
            if isinstance(w, Widget):
                w_sizing = w.sizing()
            else:
                warnings.warn(f'{w!r} is not a Widget', ColumnsWarning, stacklevel=3)
                w_sizing = frozenset((urwid.BOX, urwid.FLOW))
            if w_sizing & frozenset((Sizing.FIXED, Sizing.FLOW)):
                candidate_size = 0
                if Sizing.FIXED in w_sizing:
                    candidate_size = w.pack((), focus and i == self.focus_position)[0]
                if Sizing.FLOW in w_sizing and (not candidate_size or candidate_size > maxcol):
                    candidate_size = w.pack((maxcol,), focus and i == self.focus_position)[0]
                static_w = candidate_size
            else:
                warnings.warn(f'Unusual widget {w} sizing for {t} (box={b}). Assuming wrong sizing and using {Sizing.FLOW.upper()} for width calculation', ColumnsWarning, stacklevel=3)
                static_w = w.pack((maxcol,), focus and i == self.focus_position)[0]
        else:
            static_w = self.min_width
        if shared < static_w + self.dividechars and i > self.focus_position:
            break
        widths.append(static_w)
        shared -= static_w + self.dividechars
        if t not in {WHSettings.GIVEN, WHSettings.PACK}:
            weighted.append((width, i))
    for i, width_ in enumerate(widths):
        if shared >= 0:
            break
        shared += width_ + self.dividechars
        widths[i] = 0
        if weighted and weighted[0][1] == i:
            del weighted[0]
    if shared:
        wtotal = sum((weight for weight, i in weighted))
        grow = shared + len(weighted) * self.min_width
        for weight, i in sorted(weighted):
            width = max(int(grow * weight / wtotal + 0.5), self.min_width)
            widths[i] = width
            grow -= width
            wtotal -= weight
    self._cache_maxcol = maxcol
    self._cache_column_widths = widths
    return widths