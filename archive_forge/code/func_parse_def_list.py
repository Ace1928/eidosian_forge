import re
from ..util import strip_end
def parse_def_list(block, m, state):
    pos = m.end()
    children = list(_parse_def_item(block, m))
    m = DEF_RE.match(state.src, pos)
    while m:
        children.extend(list(_parse_def_item(block, m)))
        pos = m.end()
        m = DEF_RE.match(state.src, pos)
    state.append_token({'type': 'def_list', 'children': children})
    return pos