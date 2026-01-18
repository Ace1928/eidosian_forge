from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from ..common.utils import isStrSpace
from ..ruler import StateBase
from ..token import Token
from ..utils import EnvType
def getLines(self, begin: int, end: int, indent: int, keepLastLF: bool) -> str:
    """Cut lines range from source."""
    line = begin
    if begin >= end:
        return ''
    queue = [''] * (end - begin)
    i = 1
    while line < end:
        lineIndent = 0
        lineStart = first = self.bMarks[line]
        last = self.eMarks[line] + 1 if line + 1 < end or keepLastLF else self.eMarks[line]
        while first < last and lineIndent < indent:
            ch = self.src[first]
            if isStrSpace(ch):
                if ch == '\t':
                    lineIndent += 4 - (lineIndent + self.bsCount[line]) % 4
                else:
                    lineIndent += 1
            elif first - lineStart < self.tShift[line]:
                lineIndent += 1
            else:
                break
            first += 1
        if lineIndent > indent:
            queue[i - 1] = ' ' * (lineIndent - indent) + self.src[first:last]
        else:
            queue[i - 1] = self.src[first:last]
        line += 1
        i += 1
    return ''.join(queue)