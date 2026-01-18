from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def safe_parse_float(float_str: str) -> Any:
    float_value = parse_float(float_str)
    if isinstance(float_value, (dict, list)):
        raise ValueError('parse_float must not return dicts or lists')
    return float_value