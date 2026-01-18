from __future__ import annotations
import functools
import typing
from urwid.str_util import calc_text_pos, calc_width, get_char_width, is_wide_char, move_next_char, move_prev_char
from urwid.util import calc_trim_text, get_encoding
@functools.lru_cache(maxsize=4)
def get_ellipsis_string(encoding: str) -> str:
    """Get ellipsis character for given encoding."""
    try:
        return '…'.encode(encoding).decode(encoding)
    except UnicodeEncodeError:
        return '...'