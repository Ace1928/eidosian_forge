from __future__ import annotations
import re
from typing import TYPE_CHECKING, Callable, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def amsmath_block(state: StateBlock, startLine: int, endLine: int, silent: bool) -> bool:
    if is_code_block(state, startLine):
        return False
    begin = state.bMarks[startLine] + state.tShift[startLine]
    outcome = match_environment(state.src[begin:])
    if not outcome:
        return False
    environment, numbered, endpos = outcome
    endpos += begin
    line = startLine
    while line < endLine:
        if endpos >= state.bMarks[line] and endpos <= state.eMarks[line]:
            state.line = line + 1
            break
        line += 1
    if not silent:
        token = state.push('amsmath', 'math', 0)
        token.block = True
        token.content = state.src[begin:endpos]
        token.meta = {'environment': environment, 'numbered': numbered}
        token.map = [startLine, line]
    return True