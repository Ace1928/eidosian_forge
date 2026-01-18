from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def isPunctChar(ch: str) -> bool:
    """Check if character is a punctuation character."""
    return UNICODE_PUNCT_RE.search(ch) is not None