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
@classmethod
def from_raw_data(cls, middle: tuple[int, Widget, Hashable, int, tuple[int, int] | tuple[int] | None], top: tuple[int, Iterable[tuple[Widget, Hashable, int]]], bottom: tuple[int, Iterable[tuple[Widget, Hashable, int]]]) -> Self:
    """Construct from not typed data.

        Useful for overridden cases.
        """
    return cls(middle=VisibleInfoMiddle(*middle), top=VisibleInfoTopBottom.from_raw_data(*top), bottom=VisibleInfoTopBottom.from_raw_data(*bottom))