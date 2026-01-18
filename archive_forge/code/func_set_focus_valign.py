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
def set_focus_valign(self, valign: Literal['top', 'middle', 'bottom'] | VAlign | tuple[Literal['relative', WHSettings.RELATIVE], int]):
    """Set the focus widget's display offset and inset.

        :param valign: one of: 'top', 'middle', 'bottom' ('relative', percentage 0=top 100=bottom)
        """
    vt, va = normalize_valign(valign, ListBoxError)
    self.set_focus_valign_pending = (vt, va)