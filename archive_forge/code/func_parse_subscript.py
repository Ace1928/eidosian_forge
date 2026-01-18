import re
from ..helpers import PREVENT_BACKSLASH
def parse_subscript(inline, m, state):
    return _parse_script(inline, m, state, 'subscript')