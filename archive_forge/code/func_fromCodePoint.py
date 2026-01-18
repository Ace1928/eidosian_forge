from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def fromCodePoint(c: int) -> str:
    """Convert ordinal to unicode.

    Note, in the original Javascript two string characters were required,
    for codepoints larger than `0xFFFF`.
    But Python 3 can represent any unicode codepoint in one character.
    """
    return chr(c)