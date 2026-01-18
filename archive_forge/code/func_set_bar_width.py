from __future__ import annotations
import typing
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.util import get_encoding_mode
from .constants import BAR_SYMBOLS, Sizing
from .text import Text
from .widget import Widget, WidgetError, WidgetMeta, nocache_widget_render, nocache_widget_render_instance
def set_bar_width(self, width: int | None):
    """
        Set a preferred bar width for calculate_bar_widths to use.

        width -- width of bar or None for automatic width adjustment
        """
    if width is not None and width <= 0:
        raise ValueError(width)
    self.bar_width = width
    self._invalidate()