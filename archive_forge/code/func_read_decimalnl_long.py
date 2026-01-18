import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_decimalnl_long(f):
    """
    >>> import io

    >>> read_decimalnl_long(io.BytesIO(b"1234L\\n56"))
    1234

    >>> read_decimalnl_long(io.BytesIO(b"123456789012345678901234L\\n6"))
    123456789012345678901234
    """
    s = read_stringnl(f, decode=False, stripquotes=False)
    if s[-1:] == b'L':
        s = s[:-1]
    return int(s)