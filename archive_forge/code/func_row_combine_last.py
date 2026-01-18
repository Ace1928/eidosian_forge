from __future__ import annotations
import typing
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.util import get_encoding_mode
from .constants import BAR_SYMBOLS, Sizing
from .text import Text
from .widget import Widget, WidgetError, WidgetMeta, nocache_widget_render, nocache_widget_render_instance
def row_combine_last(count: int, row):
    o_count, o_row = o[-1]
    row = row[:]
    o_row = o_row[:]
    widget_list = []
    while row:
        (bt, w), l1, l2 = seg_combine(o_row.pop(0), row.pop(0))
        if widget_list and widget_list[-1][0] == bt:
            widget_list[-1] = (bt, widget_list[-1][1] + w)
        else:
            widget_list.append((bt, w))
        if l1:
            o_row = [l1, *o_row]
        if l2:
            row = [l2, *row]
    if o_row:
        raise BarGraphError(o_row)
    o[-1] = (o_count + count, widget_list)