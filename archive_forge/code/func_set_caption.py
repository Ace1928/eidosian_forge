from __future__ import annotations
import string
import typing
from urwid import text_layout
from urwid.canvas import CompositeCanvas
from urwid.command_map import Command
from urwid.split_repr import remove_defaults
from urwid.str_util import is_wide_char, move_next_char, move_prev_char
from urwid.util import decompose_tagmarkup
from .constants import Align, Sizing, WrapMode
from .text import Text, TextError
def set_caption(self, caption: str | tuple[Hashable, str] | list[str | tuple[Hashable, str]]) -> None:
    """
        Set the caption markup for this widget.

        :param caption: markup for caption preceding edit_text, see
                        :meth:`Text.__init__` for description of text markup.

        >>> e = Edit("")
        >>> e.set_caption("cap1")
        >>> print(e.caption)
        cap1
        >>> e.set_caption(('bold', "cap2"))
        >>> print(e.caption)
        cap2
        >>> e.attrib
        [('bold', 4)]
        >>> e.caption = "cap3"  # not supported because caption stores text but set_caption() takes markup
        Traceback (most recent call last):
        AttributeError: can't set attribute
        """
    self._caption, self._attrib = decompose_tagmarkup(caption)
    self._invalidate()