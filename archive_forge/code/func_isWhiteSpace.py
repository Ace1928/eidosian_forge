from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def isWhiteSpace(code: int) -> bool:
    """Zs (unicode class) || [\\t\\f\\v\\r\\n]"""
    if code >= 8192 and code <= 8202:
        return True
    return code in MD_WHITESPACE