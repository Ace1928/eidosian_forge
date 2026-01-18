from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def parse_key_value_pair(src: str, pos: Pos, parse_float: ParseFloat) -> tuple[Pos, Key, Any]:
    pos, key = parse_key(src, pos)
    try:
        char: str | None = src[pos]
    except IndexError:
        char = None
    if char != '=':
        raise suffixed_err(src, pos, "Expected '=' after a key in a key/value pair")
    pos += 1
    pos = skip_chars(src, pos, TOML_WS)
    pos, value = parse_value(src, pos, parse_float)
    return (pos, key, value)