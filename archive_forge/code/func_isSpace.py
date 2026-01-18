from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def isSpace(code: int | None) -> bool:
    """Check if character code is a whitespace."""
    return code in (9, 32)