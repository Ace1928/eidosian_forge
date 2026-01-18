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
def get_rows_sizes(self, size: tuple[int, int] | tuple[int] | tuple[()], focus: bool=False) -> tuple[Sequence[int], Sequence[int], Sequence[tuple[int, int] | tuple[int] | tuple[()]]]:
    """Get rows widths, heights and render size parameters"""
    if not size:
        return self._get_fixed_rows_sizes(focus=focus)
    maxcol = size[0]
    item_rows = None
    widths: list[int] = []
    heights: list[int] = []
    w_h_args: list[tuple[int, int] | tuple[int] | tuple[()]] = []
    for i, (w, (f, height)) in enumerate(self.contents):
        if isinstance(w, Widget):
            w_sizing = w.sizing()
        else:
            warnings.warn(f'{w!r} is not a Widget', PileWarning, stacklevel=3)
            w_sizing = frozenset((Sizing.FLOW, Sizing.BOX))
        item_focus = focus and self.focus == w
        widths.append(maxcol)
        if f == WHSettings.GIVEN:
            heights.append(height)
            w_h_args.append((maxcol, height))
        elif f == WHSettings.PACK or len(size) == 1:
            if Sizing.FLOW in w_sizing:
                w_h_arg: tuple[int] | tuple[()] = (maxcol,)
            elif Sizing.FIXED in w_sizing and f == WHSettings.PACK:
                w_h_arg = ()
            else:
                warnings.warn(f'Unusual widget {i} sizing {w_sizing} for {f.upper()}). Assuming wrong sizing and using {Sizing.FLOW.upper()} for height calculation', PileWarning, stacklevel=3)
                w_h_arg = (maxcol,)
            heights.append(w.pack(w_h_arg, item_focus)[1])
            w_h_args.append(w_h_arg)
        else:
            if item_rows is None:
                item_rows = self.get_item_rows(size, focus)
            rows = item_rows[i]
            heights.append(rows)
            w_h_args.append((maxcol, rows))
    return (tuple(widths), tuple(heights), tuple(w_h_args))