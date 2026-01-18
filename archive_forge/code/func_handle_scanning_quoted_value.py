from __future__ import annotations
from enum import Enum
import re
from typing import Callable
def handle_scanning_quoted_value(char: str, pos: int, tokens: TokenState) -> State:
    if char == '"':
        tokens.append(tokens.start + 1, pos, 'value')
        return State.SCANNING
    if char == '\\':
        return State.SCANNING_ESCAPED
    if char == '{' or char == '}':
        raise ParseError(f'Unexpected character whilst scanning quoted value: {char}', pos)
    if char == '\n':
        tokens.append(tokens.start + 1, pos, 'value')
        return State.SCANNING_QUOTED_VALUE
    return State.SCANNING_QUOTED_VALUE