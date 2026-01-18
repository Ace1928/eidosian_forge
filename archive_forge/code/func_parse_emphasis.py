import re
from typing import Optional, List, Dict, Any, Match
from .core import Parser, InlineState
from .util import (
from .helpers import (
def parse_emphasis(self, m: Match, state: InlineState) -> int:
    pos = m.end()
    marker = m.group(0)
    mlen = len(marker)
    if mlen == 1 and state.in_emphasis:
        state.append_token({'type': 'text', 'raw': marker})
        return pos
    elif mlen == 2 and state.in_strong:
        state.append_token({'type': 'text', 'raw': marker})
        return pos
    _end_re = EMPHASIS_END_RE[marker]
    m1 = _end_re.search(state.src, pos)
    if not m1:
        state.append_token({'type': 'text', 'raw': marker})
        return pos
    end_pos = m1.end()
    text = state.src[pos:end_pos - mlen]
    prec_pos = self.precedence_scan(m, state, end_pos)
    if prec_pos:
        return prec_pos
    new_state = state.copy()
    new_state.src = text
    if mlen == 1:
        new_state.in_emphasis = True
        children = self.render(new_state)
        state.append_token({'type': 'emphasis', 'children': children})
    elif mlen == 2:
        new_state.in_strong = True
        children = self.render(new_state)
        state.append_token({'type': 'strong', 'children': children})
    else:
        new_state.in_emphasis = True
        new_state.in_strong = True
        children = [{'type': 'strong', 'children': self.render(new_state)}]
        state.append_token({'type': 'emphasis', 'children': children})
    return end_pos