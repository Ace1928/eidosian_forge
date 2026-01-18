import re
from ..helpers import PREVENT_BACKSLASH
def parse_insert(inline, m, state):
    return _parse_to_end(inline, m, state, 'insert', _INSERT_END)