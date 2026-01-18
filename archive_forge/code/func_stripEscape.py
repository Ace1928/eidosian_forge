from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def stripEscape(string: str) -> str:
    """Strip escape \\ characters"""
    return ESCAPE_CHAR.sub('\\1', string)