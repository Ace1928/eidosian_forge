from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def isValidEntityCode(c: int) -> bool:
    if c >= 55296 and c <= 57343:
        return False
    if c >= 64976 and c <= 65007:
        return False
    if c & 65535 == 65535 or c & 65535 == 65534:
        return False
    if c >= 0 and c <= 8:
        return False
    if c == 11:
        return False
    if c >= 14 and c <= 31:
        return False
    if c >= 127 and c <= 159:
        return False
    if c > 1114111:
        return False
    return True