from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def parse_basic_str_escape(src: str, pos: Pos, *, multiline: bool=False) -> tuple[Pos, str]:
    escape_id = src[pos:pos + 2]
    pos += 2
    if multiline and escape_id in {'\\ ', '\\\t', '\\\n'}:
        if escape_id != '\\\n':
            pos = skip_chars(src, pos, TOML_WS)
            try:
                char = src[pos]
            except IndexError:
                return (pos, '')
            if char != '\n':
                raise suffixed_err(src, pos, "Unescaped '\\' in a string")
            pos += 1
        pos = skip_chars(src, pos, TOML_WS_AND_NEWLINE)
        return (pos, '')
    if escape_id == '\\u':
        return parse_hex_char(src, pos, 4)
    if escape_id == '\\U':
        return parse_hex_char(src, pos, 8)
    try:
        return (pos, BASIC_STR_ESCAPE_REPLACEMENTS[escape_id])
    except KeyError:
        raise suffixed_err(src, pos, "Unescaped '\\' in a string") from None