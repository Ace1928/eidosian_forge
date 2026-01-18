from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def escapeRE(string: str) -> str:
    string = REGEXP_ESCAPE_RE.sub('\\$&', string)
    return string