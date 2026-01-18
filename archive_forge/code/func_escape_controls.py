import re
from collections import namedtuple
def escape_controls(value):
    return _ESCAPE_CONTROLS_RE.sub(_CONTROLS_MATCH_HANDLER, value)