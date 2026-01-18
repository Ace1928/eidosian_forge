from __future__ import annotations
from collections import namedtuple
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from .._compat import DATACLASS_KWARGS
from ..common.utils import isMdAsciiPunct, isPunctChar, isWhiteSpace
from ..ruler import StateBase
from ..token import Token
from ..utils import EnvType
def scanDelims(self, start: int, canSplitWord: bool) -> Scanned:
    """
        Scan a sequence of emphasis-like markers, and determine whether
        it can start an emphasis sequence or end an emphasis sequence.

         - start - position to scan from (it should point at a valid marker);
         - canSplitWord - determine if these markers can be found inside a word

        """
    pos = start
    maximum = self.posMax
    marker = self.src[start]
    lastChar = self.src[start - 1] if start > 0 else ' '
    while pos < maximum and self.src[pos] == marker:
        pos += 1
    count = pos - start
    nextChar = self.src[pos] if pos < maximum else ' '
    isLastPunctChar = isMdAsciiPunct(ord(lastChar)) or isPunctChar(lastChar)
    isNextPunctChar = isMdAsciiPunct(ord(nextChar)) or isPunctChar(nextChar)
    isLastWhiteSpace = isWhiteSpace(ord(lastChar))
    isNextWhiteSpace = isWhiteSpace(ord(nextChar))
    left_flanking = not (isNextWhiteSpace or (isNextPunctChar and (not (isLastWhiteSpace or isLastPunctChar))))
    right_flanking = not (isLastWhiteSpace or (isLastPunctChar and (not (isNextWhiteSpace or isNextPunctChar))))
    if not canSplitWord:
        can_open = left_flanking and (not right_flanking or isLastPunctChar)
        can_close = right_flanking and (not left_flanking or isNextPunctChar)
    else:
        can_open = left_flanking
        can_close = right_flanking
    return Scanned(can_open, can_close, count)