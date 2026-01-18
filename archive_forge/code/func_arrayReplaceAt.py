from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def arrayReplaceAt(src: list[_ItemTV], pos: int, newElements: list[_ItemTV]) -> list[_ItemTV]:
    """
    Remove element from array and put another array at those position.
    Useful for some operations with tokens
    """
    return src[:pos] + newElements + src[pos + 1:]