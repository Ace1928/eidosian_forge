from __future__ import annotations
from enum import Enum
import re
from typing import Callable
def handle_start(char: str, pos: int, tokens: TokenState) -> State:
    if char == '{':
        return State.SCANNING
    raise ParseError("Attributes must start with '{'", pos)