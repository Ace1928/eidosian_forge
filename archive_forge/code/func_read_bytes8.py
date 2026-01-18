import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_bytes8(f):
    """
    >>> import io, struct, sys
    >>> read_bytes8(io.BytesIO(b"\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00abc"))
    b''
    >>> read_bytes8(io.BytesIO(b"\\x03\\x00\\x00\\x00\\x00\\x00\\x00\\x00abcdef"))
    b'abc'
    >>> bigsize8 = struct.pack("<Q", sys.maxsize//3)
    >>> read_bytes8(io.BytesIO(bigsize8 + b"abcdef"))  #doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: expected ... bytes in a bytes8, but only 6 remain
    """
    n = read_uint8(f)
    assert n >= 0
    if n > sys.maxsize:
        raise ValueError('bytes8 byte count > sys.maxsize: %d' % n)
    data = f.read(n)
    if len(data) == n:
        return data
    raise ValueError('expected %d bytes in a bytes8, but only %d remain' % (n, len(data)))