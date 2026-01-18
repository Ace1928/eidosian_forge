from __future__ import annotations
import typing
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.util import get_encoding_mode
from .constants import BAR_SYMBOLS, Sizing
from .text import Text
from .widget import Widget, WidgetError, WidgetMeta, nocache_widget_render, nocache_widget_render_instance
def hlines_display(self, disp, top: int, hlines, maxrow: int):
    """
        Add hlines to display structure represented as bar_type tuple
        values:
        (bg, 0-5)
        bg is the segment that has the hline on it
        0-5 is the hline graphic to use where 0 is a regular underscore
        and 1-5 are the UTF-8 horizontal scan line characters.
        """
    if self.use_smoothed():
        shiftr = 0
        r = [(0.2, 1), (0.4, 2), (0.6, 3), (0.8, 4), (1.0, 5)]
    else:
        shiftr = 0.5
        r = [(1.0, 0)]
    rhl = []
    for h in hlines:
        rh = float(top - h) * maxrow / top - shiftr
        if rh < 0:
            continue
        rhl.append(rh)
    hrows = []
    last_i = -1
    for rh in rhl:
        i = int(rh)
        if i == last_i:
            continue
        f = rh - i
        for spl, chnum in r:
            if f < spl:
                hrows.append((i, chnum))
                break
        last_i = i

    def fill_row(row, chnum):
        rout = []
        for bar_type, width in row:
            if isinstance(bar_type, int) and len(self.hatt) > bar_type:
                rout.append(((bar_type, chnum), width))
                continue
            rout.append((bar_type, width))
        return rout
    o = []
    k = 0
    rnum = 0
    for y_count, row in disp:
        if k >= len(hrows):
            o.append((y_count, row))
            continue
        end_block = rnum + y_count
        while k < len(hrows) and hrows[k][0] < end_block:
            i, chnum = hrows[k]
            if i - rnum > 0:
                o.append((i - rnum, row))
            o.append((1, fill_row(row, chnum)))
            rnum = i + 1
            k += 1
        if rnum < end_block:
            o.append((end_block - rnum, row))
            rnum = end_block
    return o