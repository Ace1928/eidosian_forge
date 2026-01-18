from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from ..common.utils import isStrSpace
from ..ruler import StateBase
from ..token import Token
from ..utils import EnvType
def skipCharsStr(self, pos: int, ch: str) -> int:
    """Skip character string from given position."""
    while True:
        try:
            current = self.src[pos]
        except IndexError:
            break
        if current != ch:
            break
        pos += 1
    return pos