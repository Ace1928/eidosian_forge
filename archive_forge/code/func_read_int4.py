import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_int4(f):
    """
    >>> import io
    >>> read_int4(io.BytesIO(b'\\xff\\x00\\x00\\x00'))
    255
    >>> read_int4(io.BytesIO(b'\\x00\\x00\\x00\\x80')) == -(2**31)
    True
    """
    data = f.read(4)
    if len(data) == 4:
        return _unpack('<i', data)[0]
    raise ValueError('not enough data in stream to read int4')