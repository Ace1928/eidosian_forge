from __future__ import annotations
from enum import Enum
import re
from typing import Callable
def handle_scanning(char: str, pos: int, tokens: TokenState) -> State:
    if char == ' ' or char == '\t' or char == '\n' or (char == '\r'):
        return State.SCANNING
    if char == '}':
        return State.DONE
    if char == '#':
        tokens.set_start(pos)
        return State.SCANNING_ID
    if char == '%':
        tokens.set_start(pos)
        return State.SCANNING_COMMENT
    if char == '.':
        tokens.set_start(pos)
        return State.SCANNING_CLASS
    if REGEX_KEY_CHARACTERS.fullmatch(char):
        tokens.set_start(pos)
        return State.SCANNING_KEY
    raise ParseError(f'Unexpected character whilst scanning: {char}', pos)