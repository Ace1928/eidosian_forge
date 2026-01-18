from __future__ import annotations
import typing
import warnings
from urwid.split_repr import remove_defaults
from .columns import Columns
from .constants import Align, Sizing, WHSettings
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin
from .divider import Divider
from .monitored_list import MonitoredFocusList, MonitoredList
from .padding import Padding
from .pile import Pile
from .widget import Widget, WidgetError, WidgetWarning, WidgetWrap
def generate_display_widget(self, size: tuple[int] | tuple[()]) -> Divider | Pile:
    """
        Actually generate display widget (ignoring cache)
        """
    maxcol = self._get_maxcol(size)
    divider = Divider()
    if not self.contents:
        return divider
    if self.v_sep > 1:
        divider.top = self.v_sep - 1
    c = None
    p = Pile([])
    used_space = 0
    for i, (w, (_width_type, width_amount)) in enumerate(self.contents):
        if c is None or maxcol - used_space < width_amount:
            if self.v_sep:
                p.contents.append((divider, p.options()))
            c = Columns([], self.h_sep)
            column_focused = False
            pad = Padding(c, self.align)
            pad.first_position = i
            p.contents.append((pad, p.options()))
        c.contents.append((w, c.options(WHSettings.GIVEN, min(width_amount, maxcol))))
        if i == self.focus_position or (not column_focused and w.selectable()):
            c.focus_position = len(c.contents) - 1
            column_focused = True
        if i == self.focus_position:
            p.focus_position = len(p.contents) - 1
        used_space = sum((x[1][1] for x in c.contents)) + self.h_sep * len(c.contents)
        pad.width = used_space - self.h_sep
    if self.v_sep:
        del p.contents[:1]
    else:
        p._contents_modified()
    return p