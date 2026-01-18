from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def parse_literal_str(src: str, pos: Pos) -> tuple[Pos, str]:
    pos += 1
    start_pos = pos
    pos = skip_until(src, pos, "'", error_on=ILLEGAL_LITERAL_STR_CHARS, error_on_eof=True)
    return (pos + 1, src[start_pos:pos])