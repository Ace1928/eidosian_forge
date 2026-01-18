from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, Callable, Match, Sequence, TypedDict
from markdown_it import MarkdownIt
from markdown_it.common.utils import charCodeAt
def make_block_func(rule: RuleDictType) -> Callable[[StateBlock, int, int, bool], bool]:

    def _func(state: StateBlock, begLine: int, endLine: int, silent: bool) -> bool:
        begin = state.bMarks[begLine] + state.tShift[begLine]
        res = applyRule(rule, state.src, begin, state.parentType == 'blockquote')
        if res:
            if not silent:
                token = state.push(rule['name'], 'math', 0)
                token.block = True
                token.content = res[1]
                token.info = res[len(res.groups())]
                token.markup = rule['tag']
            line = begLine
            endpos = begin + res.end() - 1
            while line < endLine:
                if endpos >= state.bMarks[line] and endpos <= state.eMarks[line]:
                    state.line = line + 1
                    break
                line += 1
        return bool(res)
    return _func