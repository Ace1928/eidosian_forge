from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def parse_array(src: str, pos: Pos, parse_float: ParseFloat) -> tuple[Pos, list]:
    pos += 1
    array: list = []
    pos = skip_comments_and_array_ws(src, pos)
    if src.startswith(']', pos):
        return (pos + 1, array)
    while True:
        pos, val = parse_value(src, pos, parse_float)
        array.append(val)
        pos = skip_comments_and_array_ws(src, pos)
        c = src[pos:pos + 1]
        if c == ']':
            return (pos + 1, array)
        if c != ',':
            raise suffixed_err(src, pos, 'Unclosed array')
        pos += 1
        pos = skip_comments_and_array_ws(src, pos)
        if src.startswith(']', pos):
            return (pos + 1, array)