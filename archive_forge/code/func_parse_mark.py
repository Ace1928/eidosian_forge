import re
from ..helpers import PREVENT_BACKSLASH
def parse_mark(inline, m, state):
    return _parse_to_end(inline, m, state, 'mark', _MARK_END)