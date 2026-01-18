from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def parse_basic_str_escape_multiline(src: str, pos: Pos) -> tuple[Pos, str]:
    return parse_basic_str_escape(src, pos, multiline=True)