from __future__ import division
import re
import stat
from .helpers import (
def format_path(p, quote_spaces=False):
    """Format a path in utf8, quoting it if necessary."""
    if b'\n' in p:
        p = re.sub(b'\n', b'\\n', p)
        quote = True
    else:
        quote = p[0] == b'"' or (quote_spaces and b' ' in p)
    if quote:
        extra = GIT_FAST_IMPORT_NEEDS_EXTRA_SPACE_AFTER_QUOTE and b' ' or b''
        p = b'"' + p + b'"' + extra
    return p