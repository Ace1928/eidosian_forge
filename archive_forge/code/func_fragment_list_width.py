from __future__ import annotations
from typing import Iterable, cast
from prompt_toolkit.utils import get_cwidth
from .base import (
def fragment_list_width(fragments: StyleAndTextTuples) -> int:
    """
    Return the character width of this text fragment list.
    (Take double width characters into account.)

    :param fragments: List of ``(style_str, text)`` or
        ``(style_str, text, mouse_handler)`` tuples.
    """
    ZeroWidthEscape = '[ZeroWidthEscape]'
    return sum((get_cwidth(c) for item in fragments for c in item[1] if ZeroWidthEscape not in item[0]))