from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, Callable, Match, Sequence, TypedDict
from markdown_it import MarkdownIt
from markdown_it.common.utils import charCodeAt
def make_inline_func(rule: RuleDictType) -> Callable[[StateInline, bool], bool]:

    def _func(state: StateInline, silent: bool) -> bool:
        res = applyRule(rule, state.src, state.pos, False)
        if res:
            if not silent:
                token = state.push(rule['name'], 'math', 0)
                token.content = res[1]
                token.markup = rule['tag']
            state.pos += res.end()
        return bool(res)
    return _func