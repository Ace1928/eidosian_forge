from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def diff_bytes(dfunc, a, b, fromfile=b'', tofile=b'', fromfiledate=b'', tofiledate=b'', n=3, lineterm=b'\n'):
    """
    Compare `a` and `b`, two sequences of lines represented as bytes rather
    than str. This is a wrapper for `dfunc`, which is typically either
    unified_diff() or context_diff(). Inputs are losslessly converted to
    strings so that `dfunc` only has to worry about strings, and encoded
    back to bytes on return. This is necessary to compare files with
    unknown or inconsistent encoding. All other inputs (except `n`) must be
    bytes rather than str.
    """

    def decode(s):
        try:
            return s.decode('ascii', 'surrogateescape')
        except AttributeError as err:
            msg = 'all arguments must be bytes, not %s (%r)' % (type(s).__name__, s)
            raise TypeError(msg) from err
    a = list(map(decode, a))
    b = list(map(decode, b))
    fromfile = decode(fromfile)
    tofile = decode(tofile)
    fromfiledate = decode(fromfiledate)
    tofiledate = decode(tofiledate)
    lineterm = decode(lineterm)
    lines = dfunc(a, b, fromfile, tofile, fromfiledate, tofiledate, n, lineterm)
    for line in lines:
        yield line.encode('ascii', 'surrogateescape')