import re
import sys
from ..errors import BzrError
from ..filters import ContentFilter
def eol_lookup(key):
    filter = _eol_filter_stack_map.get(key)
    if filter is None:
        raise BzrError("Unknown eol value '%s'" % key)
    return filter