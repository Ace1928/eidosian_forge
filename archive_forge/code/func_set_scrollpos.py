from __future__ import annotations
import contextlib
import enum
import typing
from typing_extensions import Protocol, runtime_checkable
from .constants import BOX_SYMBOLS, SHADE_SYMBOLS, Sizing
from .widget_decoration import WidgetDecoration, WidgetError
def set_scrollpos(self, position: typing.SupportsInt) -> None:
    """Set scrolling position

        If `position` is positive it is interpreted as lines from the top.
        If `position` is negative it is interpreted as lines from the bottom.

        Values that are too high or too low values are automatically adjusted during rendering.
        """
    self._trim_top = int(position)
    self._invalidate()