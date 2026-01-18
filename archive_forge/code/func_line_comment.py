from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def line_comment(state: StateBlock, startLine: int, endLine: int, silent: bool) -> bool:
    if is_code_block(state, startLine):
        return False
    pos = state.bMarks[startLine] + state.tShift[startLine]
    maximum = state.eMarks[startLine]
    if state.src[pos] != '%':
        return False
    if silent:
        return True
    token = state.push('myst_line_comment', '', 0)
    token.attrSet('class', 'myst-line-comment')
    token.content = state.src[pos + 1:maximum].rstrip()
    token.markup = '%'
    for nextLine in itertools.count(startLine + 1):
        if nextLine >= endLine:
            break
        pos = state.bMarks[nextLine] + state.tShift[nextLine]
        maximum = state.eMarks[nextLine]
        if state.src[pos] != '%':
            break
        token.content += '\n' + state.src[pos + 1:maximum].rstrip()
    state.line = nextLine
    token.map = [startLine, nextLine]
    return True