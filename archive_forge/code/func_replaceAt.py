from __future__ import annotations
import re
from typing import Any
from ..common.utils import charCodeAt, isMdAsciiPunct, isPunctChar, isWhiteSpace
from ..token import Token
from .state_core import StateCore
def replaceAt(string: str, index: int, ch: str) -> str:
    assert index >= 0
    return string[:index] + ch + string[index + 1:]