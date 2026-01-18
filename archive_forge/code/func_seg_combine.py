from __future__ import annotations
import typing
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.util import get_encoding_mode
from .constants import BAR_SYMBOLS, Sizing
from .text import Text
from .widget import Widget, WidgetError, WidgetMeta, nocache_widget_render, nocache_widget_render_instance
def seg_combine(a, b):
    (bt1, w1), (bt2, w2) = (a, b)
    if (bt1, w1) == (bt2, w2):
        return ((bt1, w1), None, None)
    wmin = min(w1, w2)
    l1 = l2 = None
    if w1 > w2:
        l1 = (bt1, w1 - w2)
    elif w2 > w1:
        l2 = (bt2, w2 - w1)
    if isinstance(bt1, tuple):
        return ((bt1, wmin), l1, l2)
    if (bt2, bt1) not in self.satt:
        if r < 4:
            return ((bt2, wmin), l1, l2)
        return ((bt1, wmin), l1, l2)
    return (((bt2, bt1, 8 - r), wmin), l1, l2)