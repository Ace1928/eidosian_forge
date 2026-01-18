from __future__ import annotations
import contextlib
import enum
import typing
from typing_extensions import Protocol, runtime_checkable
from .constants import BOX_SYMBOLS, SHADE_SYMBOLS, Sizing
from .widget_decoration import WidgetDecoration, WidgetError
def orig_iter(w: Widget) -> Iterator[Widget]:
    while hasattr(w, 'original_widget'):
        w = w.original_widget
        yield w
    yield w