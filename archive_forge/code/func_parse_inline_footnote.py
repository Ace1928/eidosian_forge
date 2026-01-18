import re
from ..core import BlockState
from ..util import unikey
from ..helpers import LINK_LABEL
def parse_inline_footnote(inline, m: re.Match, state):
    key = unikey(m.group('footnote_key'))
    ref = state.env.get('ref_footnotes')
    if ref and key in ref:
        notes = state.env.get('footnotes')
        if not notes:
            notes = []
        if key not in notes:
            notes.append(key)
            state.env['footnotes'] = notes
        state.append_token({'type': 'footnote_ref', 'raw': key, 'attrs': {'index': notes.index(key) + 1}})
    else:
        state.append_token({'type': 'text', 'raw': m.group(0)})
    return m.end()