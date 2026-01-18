from __future__ import annotations
import typing
import warnings
from urwid.canvas import CanvasOverlay, CompositeCanvas
from urwid.split_repr import remove_defaults
from .constants import (
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin
from .filler import calculate_top_bottom_filler
from .padding import calculate_left_right_padding
from .widget import Widget, WidgetError, WidgetWarning
def calculate_padding_filler(self, size: tuple[int, int], focus: bool) -> tuple[int, int, int, int]:
    """Return (padding left, right, filler top, bottom)."""
    maxcol, maxrow = size
    height = None
    if self.width_type == WHSettings.PACK:
        width, height = self.top_w.pack((), focus=focus)
        if not height:
            raise OverlayError('fixed widget must have a height')
        left, right = calculate_left_right_padding(maxcol, self.align_type, self.align_amount, WrapMode.CLIP, width, None, self.left, self.right)
    else:
        left, right = calculate_left_right_padding(maxcol, self.align_type, self.align_amount, self.width_type, self.width_amount, self.min_width, self.left, self.right)
    if height:
        top, bottom = calculate_top_bottom_filler(maxrow, self.valign_type, self.valign_amount, WHSettings.GIVEN, height, None, self.top, self.bottom)
        if maxrow - top - bottom < height:
            bottom = maxrow - top - height
    elif self.height_type == WHSettings.PACK:
        height = self.top_w.rows((maxcol,), focus=focus)
        top, bottom = calculate_top_bottom_filler(maxrow, self.valign_type, self.valign_amount, WHSettings.GIVEN, height, None, self.top, self.bottom)
        if height > maxrow:
            bottom = maxrow - height
    else:
        top, bottom = calculate_top_bottom_filler(maxrow, self.valign_type, self.valign_amount, self.height_type, self.height_amount, self.min_height, self.top, self.bottom)
    return (left, right, top, bottom)