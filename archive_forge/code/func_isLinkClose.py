from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def isLinkClose(string: str) -> bool:
    return bool(LINK_CLOSE_RE.search(string))