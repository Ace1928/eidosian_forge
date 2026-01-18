import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_string4(f):
    """
    >>> import io
    >>> read_string4(io.BytesIO(b"\\x00\\x00\\x00\\x00abc"))
    ''
    >>> read_string4(io.BytesIO(b"\\x03\\x00\\x00\\x00abcdef"))
    'abc'
    >>> read_string4(io.BytesIO(b"\\x00\\x00\\x00\\x03abcdef"))
    Traceback (most recent call last):
    ...
    ValueError: expected 50331648 bytes in a string4, but only 6 remain
    """
    n = read_int4(f)
    if n < 0:
        raise ValueError('string4 byte count < 0: %d' % n)
    data = f.read(n)
    if len(data) == n:
        return data.decode('latin-1')
    raise ValueError('expected %d bytes in a string4, but only %d remain' % (n, len(data)))