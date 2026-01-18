from __future__ import annotations
import typing
from urwid import text_layout
from urwid.canvas import apply_text_layout
from urwid.split_repr import remove_defaults
from urwid.str_util import calc_width
from urwid.util import decompose_tagmarkup, get_encoding
from .constants import Align, Sizing, WrapMode
from .widget import Widget, WidgetError
@property
def attrib(self) -> list[tuple[Hashable, int]]:
    """
        Read-only property returning the run-length encoded display
        attributes of this widget
        """
    return self.get_text()[1]