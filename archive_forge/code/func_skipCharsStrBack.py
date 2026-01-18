from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from ..common.utils import isStrSpace
from ..ruler import StateBase
from ..token import Token
from ..utils import EnvType
def skipCharsStrBack(self, pos: int, ch: str, minimum: int) -> int:
    """Skip character string reverse from given position - 1."""
    if pos <= minimum:
        return pos
    while pos > minimum:
        pos -= 1
        if ch != self.src[pos]:
            return pos + 1
    return pos