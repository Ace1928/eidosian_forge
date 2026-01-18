from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def parse_multiline_str(src: str, pos: Pos, *, literal: bool) -> tuple[Pos, str]:
    pos += 3
    if src.startswith('\n', pos):
        pos += 1
    if literal:
        delim = "'"
        end_pos = skip_until(src, pos, "'''", error_on=ILLEGAL_MULTILINE_LITERAL_STR_CHARS, error_on_eof=True)
        result = src[pos:end_pos]
        pos = end_pos + 3
    else:
        delim = '"'
        pos, result = parse_basic_str(src, pos, multiline=True)
    if not src.startswith(delim, pos):
        return (pos, result)
    pos += 1
    if not src.startswith(delim, pos):
        return (pos, result + delim)
    pos += 1
    return (pos, result + delim * 2)