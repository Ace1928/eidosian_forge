import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_bytes1(f):
    """
    >>> import io
    >>> read_bytes1(io.BytesIO(b"\\x00"))
    b''
    >>> read_bytes1(io.BytesIO(b"\\x03abcdef"))
    b'abc'
    """
    n = read_uint1(f)
    assert n >= 0
    data = f.read(n)
    if len(data) == n:
        return data
    raise ValueError('expected %d bytes in a bytes1, but only %d remain' % (n, len(data)))