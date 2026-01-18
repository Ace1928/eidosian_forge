from __future__ import annotations
import contextlib
import enum
import typing
from typing_extensions import Protocol, runtime_checkable
from .constants import BOX_SYMBOLS, SHADE_SYMBOLS, Sizing
from .widget_decoration import WidgetDecoration, WidgetError
@property
def scrolling_base_widget(self) -> SupportsScroll | SupportsRelativeScroll:
    """Nearest `original_widget` that is compatible with the scrolling API"""

    def orig_iter(w: Widget) -> Iterator[Widget]:
        while hasattr(w, 'original_widget'):
            w = w.original_widget
            yield w
        yield w
    w = self
    for w in orig_iter(self):
        if isinstance(w, SupportsScroll):
            return w
    raise ScrollableError(f'Not compatible to be wrapped by ScrollBar: {w!r}')