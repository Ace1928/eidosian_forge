import re
from ..core import BlockState
from ..util import unikey
from ..helpers import LINK_LABEL
def parse_ref_footnote(block, m: re.Match, state: BlockState):
    ref = state.env.get('ref_footnotes')
    if not ref:
        ref = {}
    key = unikey(m.group('footnote_key'))
    if key not in ref:
        ref[key] = m.group('footnote_text')
        state.env['ref_footnotes'] = ref
    return m.end()