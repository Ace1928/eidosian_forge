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
def set_edit_text(self, text: str) -> None:
    """
        Set the edit text for this widget.

        :param text: text for editing, type (bytes or unicode)
                     must match the text in the caption
        :type text: bytes or unicode

        >>> e = Edit()
        >>> e.set_edit_text(u"yes")
        >>> print(e.edit_text)
        yes
        >>> e
        <Edit selectable flow widget 'yes' edit_pos=0>
        >>> e.edit_text = u"no"  # Urwid 0.9.9 or later
        >>> print(e.edit_text)
        no
        """
    text = self._normalize_to_caption(text)
    self.highlight = None
    self._emit('change', text)
    old_text = self._edit_text
    self._edit_text = text
    self.edit_pos = min(self.edit_pos, len(text))
    self._emit('postchange', old_text)
    self._invalidate()