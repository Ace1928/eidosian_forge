from __future__ import annotations
import typing
import warnings
from urwid.canvas import CompositeCanvas
from urwid.split_repr import remove_defaults
from urwid.util import int_scale
from .constants import (
from .widget_decoration import WidgetDecoration, WidgetError
def filler_values(self, size: tuple[int, int] | tuple[int], focus: bool) -> tuple[int, int]:
    """
        Return the number of rows to pad on the top and bottom.

        Override this method to define custom padding behaviour.
        """
    maxcol, maxrow = self.pack(size, focus)
    if self.height_type == WHSettings.PACK:
        height = self._original_widget.rows((maxcol,), focus=focus)
        return calculate_top_bottom_filler(maxrow, self.valign_type, self.valign_amount, WHSettings.GIVEN, height, None, self.top, self.bottom)
    return calculate_top_bottom_filler(maxrow, self.valign_type, self.valign_amount, self.height_type, self.height_amount, self.min_height, self.top, self.bottom)