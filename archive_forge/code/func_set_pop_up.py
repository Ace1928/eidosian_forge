from __future__ import annotations
import contextlib
import dataclasses
import typing
import warnings
import weakref
from contextlib import suppress
from urwid.str_util import calc_text_pos, calc_width
from urwid.text_layout import LayoutSegment, trim_line
from urwid.util import (
def set_pop_up(self, w: Widget, left: int, top: int, overlay_width: int, overlay_height: int) -> None:
    """
        This method adds pop-up information to the canvas.  This information
        is intercepted by a PopUpTarget widget higher in the chain to
        display a pop-up at the given (left, top) position relative to the
        current canvas.

        :param w: widget to use for the pop-up
        :type w: widget
        :param left: x position for left edge of pop-up >= 0
        :type left: int
        :param top: y position for top edge of pop-up >= 0
        :type top: int
        :param overlay_width: width of overlay in screen columns > 0
        :type overlay_width: int
        :param overlay_height: height of overlay in screen rows > 0
        :type overlay_height: int
        """
    if self.widget_info and self.cacheable:
        raise self._finalized_error
    self.coords['pop up'] = (left, top, (w, overlay_width, overlay_height))