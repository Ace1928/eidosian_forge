from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def block_break(state: StateBlock, startLine: int, endLine: int, silent: bool) -> bool:
    if is_code_block(state, startLine):
        return False
    pos = state.bMarks[startLine] + state.tShift[startLine]
    maximum = state.eMarks[startLine]
    marker = state.src[pos]
    pos += 1
    if marker != '+':
        return False
    cnt = 1
    while pos < maximum:
        ch = state.src[pos]
        if ch != marker and ch not in ('\t', ' '):
            break
        if ch == marker:
            cnt += 1
        pos += 1
    if cnt < 3:
        return False
    if silent:
        return True
    state.line = startLine + 1
    token = state.push('myst_block_break', 'hr', 0)
    token.attrSet('class', 'myst-block')
    token.content = state.src[pos:maximum].strip()
    token.map = [startLine, state.line]
    token.markup = marker * cnt
    return True