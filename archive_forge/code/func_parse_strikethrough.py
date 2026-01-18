import re
from ..helpers import PREVENT_BACKSLASH
def parse_strikethrough(inline, m, state):
    return _parse_to_end(inline, m, state, 'strikethrough', _STRIKE_END)