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
def insert_text_result(self, text: str) -> tuple[str | bytes, int]:
    """
        Return result of insert_text(text) without actually performing the
        insertion.  Handy for pre-validation.

        :param text: text for inserting, type (bytes or unicode)
                     must match the text in the caption
        :type text: bytes or unicode
        """
    text = self._normalize_to_caption(text)
    if self.highlight:
        start, stop = self.highlight
        btext, etext = (self.edit_text[:start], self.edit_text[stop:])
        result_text = btext + etext
        result_pos = start
    else:
        result_text = self.edit_text
        result_pos = self.edit_pos
    try:
        result_text = result_text[:result_pos] + text + result_text[result_pos:]
    except (IndexError, TypeError) as exc:
        raise ValueError(repr((self.edit_text, result_text, text))).with_traceback(exc.__traceback__) from exc
    result_pos += len(text)
    return (result_text, result_pos)