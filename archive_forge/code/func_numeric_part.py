import re
from . import cursors, _mysql
from ._exceptions import (
def numeric_part(s):
    """Returns the leading numeric part of a string.

    >>> numeric_part("20-alpha")
    20
    >>> numeric_part("foo")
    >>> numeric_part("16b")
    16
    """
    m = re_numeric_part.match(s)
    if m:
        return int(m.group(1))
    return None