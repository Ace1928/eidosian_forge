from __future__ import annotations
from enum import Enum
import re
from typing import Callable
def handle_scanning_class(char: str, pos: int, tokens: TokenState) -> State:
    if not REGEX_SPACE_PUNCTUATION.fullmatch(char):
        return State.SCANNING_CLASS
    if char == '}':
        if pos - 1 > tokens.start:
            tokens.append(tokens.start + 1, pos, 'class')
        return State.DONE
    if REGEX_SPACE.fullmatch(char):
        if pos - 1 > tokens.start:
            tokens.append(tokens.start + 1, pos, 'class')
        return State.SCANNING
    raise ParseError(f'Unexpected character whilst scanning class: {char}', pos)