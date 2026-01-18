import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_bytes4(f):
    """
    >>> import io
    >>> read_bytes4(io.BytesIO(b"\\x00\\x00\\x00\\x00abc"))
    b''
    >>> read_bytes4(io.BytesIO(b"\\x03\\x00\\x00\\x00abcdef"))
    b'abc'
    >>> read_bytes4(io.BytesIO(b"\\x00\\x00\\x00\\x03abcdef"))
    Traceback (most recent call last):
    ...
    ValueError: expected 50331648 bytes in a bytes4, but only 6 remain
    """
    n = read_uint4(f)
    assert n >= 0
    if n > sys.maxsize:
        raise ValueError('bytes4 byte count > sys.maxsize: %d' % n)
    data = f.read(n)
    if len(data) == n:
        return data
    raise ValueError('expected %d bytes in a bytes4, but only %d remain' % (n, len(data)))