from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Sequence
from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def admonition(state: StateBlock, startLine: int, endLine: int, silent: bool) -> bool:
    if is_code_block(state, startLine):
        return False
    start = state.bMarks[startLine] + state.tShift[startLine]
    maximum = state.eMarks[startLine]
    if state.src[start] not in MARKER_CHARS:
        return False
    marker = ''
    marker_len = MAX_MARKER_LEN
    while marker_len > 0:
        marker_pos = start + marker_len
        markup = state.src[start:marker_pos]
        if markup in MARKERS:
            marker = markup
            break
        marker_len -= 1
    else:
        return False
    params = state.src[marker_pos:maximum]
    if not _validate(params):
        return False
    if silent:
        return True
    old_parent = state.parentType
    old_line_max = state.lineMax
    old_indent = state.blkIndent
    blk_start = marker_pos
    while blk_start < maximum and state.src[blk_start] == ' ':
        blk_start += 1
    state.parentType = 'admonition'
    marker_alignment_correction = MARKER_LEN - len(marker)
    state.blkIndent += blk_start - start + marker_alignment_correction
    was_empty = False
    next_line = startLine
    while True:
        next_line += 1
        if next_line >= endLine:
            break
        pos = state.bMarks[next_line] + state.tShift[next_line]
        maximum = state.eMarks[next_line]
        is_empty = state.sCount[next_line] < state.blkIndent
        if is_empty and was_empty:
            break
        was_empty = is_empty
        if pos < maximum and state.sCount[next_line] < state.blkIndent:
            break
    state.lineMax = next_line
    tag, title = _get_tag(params)
    token = state.push('admonition_open', 'div', 1)
    token.markup = markup
    token.block = True
    token.attrs = {'class': ' '.join(['admonition', tag, *_extra_classes(markup)])}
    token.meta = {'tag': tag}
    token.content = title
    token.info = params
    token.map = [startLine, next_line]
    if title:
        title_markup = f'{markup} {tag}'
        token = state.push('admonition_title_open', 'p', 1)
        token.markup = title_markup
        token.attrs = {'class': 'admonition-title'}
        token.map = [startLine, startLine + 1]
        token = state.push('inline', '', 0)
        token.content = title
        token.map = [startLine, startLine + 1]
        token.children = []
        token = state.push('admonition_title_close', 'p', -1)
    state.md.block.tokenize(state, startLine + 1, next_line)
    token = state.push('admonition_close', 'div', -1)
    token.markup = markup
    token.block = True
    state.parentType = old_parent
    state.lineMax = old_line_max
    state.blkIndent = old_indent
    state.line = next_line
    return True