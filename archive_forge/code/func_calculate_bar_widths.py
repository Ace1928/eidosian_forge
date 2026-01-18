from __future__ import annotations
import typing
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.util import get_encoding_mode
from .constants import BAR_SYMBOLS, Sizing
from .text import Text
from .widget import Widget, WidgetError, WidgetMeta, nocache_widget_render, nocache_widget_render_instance
def calculate_bar_widths(self, size: tuple[int, int], bardata):
    """
        Return a list of bar widths, one for each bar in data.

        If self.bar_width is None this implementation will stretch
        the bars across the available space specified by maxcol.
        """
    maxcol, _maxrow = size
    if self.bar_width is not None:
        return [self.bar_width] * min(len(bardata), maxcol // self.bar_width)
    if len(bardata) >= maxcol:
        return [1] * maxcol
    widths = []
    grow = maxcol
    remain = len(bardata)
    for _row in bardata:
        w = int(float(grow) / remain + 0.5)
        widths.append(w)
        grow -= w
        remain -= 1
    return widths