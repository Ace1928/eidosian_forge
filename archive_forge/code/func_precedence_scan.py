import re
from typing import Optional, List, Dict, Any, Match
from .core import Parser, InlineState
from .util import (
from .helpers import (
def precedence_scan(self, m: Match, state: InlineState, end_pos: int, rules=None):
    if rules is None:
        rules = ['codespan', 'link', 'prec_auto_link', 'prec_inline_html']
    mark_pos = m.end()
    sc = self.compile_sc(rules)
    m1 = sc.search(state.src, mark_pos, end_pos)
    if not m1:
        return
    rule_name = m1.lastgroup.replace('prec_', '')
    sc = self.compile_sc([rule_name])
    m2 = sc.match(state.src, m1.start())
    if not m2:
        return
    func = self._methods[rule_name]
    new_state = state.copy()
    new_state.src = state.src
    m2_pos = func(m2, new_state)
    if not m2_pos or m2_pos < end_pos:
        return
    raw_text = state.src[m.start():m2.start()]
    state.append_token({'type': 'text', 'raw': raw_text})
    for token in new_state.tokens:
        state.append_token(token)
    return m2_pos