from __future__ import absolute_import
import re
import sys
import struct
from .compat import PY3, unichr
from .scanner import make_scanner, JSONDecodeError
def bounded_int(s, INT_MAX_STR_DIGITS=4300):
    """Backport of the integer string length conversion limitation

        https://docs.python.org/3/library/stdtypes.html#int-max-str-digits
        """
    if len(s) > INT_MAX_STR_DIGITS:
        raise ValueError('Exceeds the limit (%s) for integer string conversion: value has %s digits' % (INT_MAX_STR_DIGITS, len(s)))
    return int(s)