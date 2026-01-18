from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def replacer_func(match: Match[str]) -> str:
    escaped = match.group(1)
    if escaped:
        return escaped
    entity = match.group(2)
    return replaceEntityPattern(match.group(), entity)