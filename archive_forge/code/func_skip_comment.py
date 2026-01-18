from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def skip_comment(src: str, pos: Pos) -> Pos:
    try:
        char: str | None = src[pos]
    except IndexError:
        char = None
    if char == '#':
        return skip_until(src, pos + 1, '\n', error_on=ILLEGAL_COMMENT_CHARS, error_on_eof=False)
    return pos