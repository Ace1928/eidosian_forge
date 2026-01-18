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
def get_item_rows(self, size: tuple[()] | tuple[int] | tuple[int, int], focus: bool) -> list[int]:
    """
        Return a list of the number of rows used by each widget in self.contents
        """
    remaining = None
    maxcol = size[0]
    if len(size) == 2:
        remaining = size[1]
    rows_numbers = []
    if remaining is None:
        for i, (w, (f, height)) in enumerate(self.contents):
            if isinstance(w, Widget):
                w_sizing = w.sizing()
            else:
                warnings.warn(f'{w!r} is not a Widget', PileWarning, stacklevel=3)
                w_sizing = frozenset((Sizing.FLOW, Sizing.BOX))
            focused = focus and self.focus == w
            if f == WHSettings.GIVEN:
                rows_numbers.append(height)
            elif Sizing.FLOW in w_sizing:
                rows_numbers.append(w.rows((maxcol,), focus=focused))
            elif Sizing.FIXED in w_sizing and f == WHSettings.PACK:
                rows_numbers.append(w.pack((), focused)[0])
            else:
                warnings.warn(f'Unusual widget {i} sizing {w_sizing} for {f.upper()}). Assuming wrong sizing and using {Sizing.FLOW.upper()} for height calculation', PileWarning, stacklevel=3)
                rows_numbers.append(w.rows((maxcol,), focus=focused))
        return rows_numbers
    wtotal = 0
    for w, (f, height) in self.contents:
        if f == WHSettings.PACK:
            rows = w.rows((maxcol,), focus=focus and self.focus == w)
            rows_numbers.append(rows)
            remaining -= rows
        elif f == WHSettings.GIVEN:
            rows_numbers.append(height)
            remaining -= height
        elif height:
            rows_numbers.append(None)
            wtotal += height
        else:
            rows_numbers.append(0)
    if wtotal == 0:
        raise PileError('No weighted widgets found for Pile treated as a box widget')
    remaining = max(remaining, 0)
    for i, (_w, (_f, height)) in enumerate(self.contents):
        li = rows_numbers[i]
        if li is None:
            rows = int(float(remaining) * height / wtotal + 0.5)
            rows_numbers[i] = rows
            remaining -= rows
            wtotal -= height
    return rows_numbers