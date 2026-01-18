from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from ..common.utils import isStrSpace
from ..ruler import StateBase
from ..token import Token
from ..utils import EnvType
def skipSpacesBack(self, pos: int, minimum: int) -> int:
    """Skip spaces from given position in reverse."""
    if pos <= minimum:
        return pos
    while pos > minimum:
        pos -= 1
        if not isStrSpace(self.src[pos]):
            return pos + 1
    return pos