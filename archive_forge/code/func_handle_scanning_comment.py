from __future__ import annotations
from enum import Enum
import re
from typing import Callable
def handle_scanning_comment(char: str, pos: int, tokens: TokenState) -> State:
    if char == '%':
        return State.SCANNING
    return State.SCANNING_COMMENT