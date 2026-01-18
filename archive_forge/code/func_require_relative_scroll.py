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
def require_relative_scroll(self, size: tuple[int, int], focus: bool=False) -> bool:
    """Widget require relative scroll due to performance limitations of real lines count calculation."""
    return isinstance(self._body, (Sized, EstimatedSized)) and size[1] * 3 < operator.length_hint(self.body)