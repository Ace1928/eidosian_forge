import re
from ..util import unikey
from ..helpers import parse_link, parse_link_label
def parse_ruby(inline, m, state):
    text = m.group(0)[1:-2]
    items = text.split(')')
    tokens = []
    for item in items:
        rb, rt = item.split('(')
        tokens.append({'type': 'ruby', 'raw': rb, 'attrs': {'rt': rt}})
    end_pos = m.end()
    next_match = _ruby_re.match(state.src, end_pos)
    if next_match:
        for tok in tokens:
            state.append_token(tok)
        return parse_ruby(inline, next_match, state)
    if end_pos < len(state.src):
        link_pos = _parse_ruby_link(inline, state, end_pos, tokens)
        if link_pos:
            return link_pos
    for tok in tokens:
        state.append_token(tok)
    return end_pos