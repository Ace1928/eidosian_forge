import math
import sys
import struct
from Cryptodome import Random
from Cryptodome.Util.py3compat import iter_range
import struct
import warnings
def long_to_bytes(n, blocksize=0):
    """Convert a positive integer to a byte string using big endian encoding.

    If :data:`blocksize` is absent or zero, the byte string will
    be of minimal length.

    Otherwise, the length of the byte string is guaranteed to be a multiple
    of :data:`blocksize`. If necessary, zeroes (``\\x00``) are added at the left.

    .. note::
        In Python 3, if you are sure that :data:`n` can fit into
        :data:`blocksize` bytes, you can simply use the native method instead::

            >>> n.to_bytes(blocksize, 'big')

        For instance::

            >>> n = 80
            >>> n.to_bytes(2, 'big')
            b'\\x00P'

        However, and unlike this ``long_to_bytes()`` function,
        an ``OverflowError`` exception is raised if :data:`n` does not fit.
    """
    if n < 0 or blocksize < 0:
        raise ValueError('Values must be non-negative')
    result = []
    pack = struct.pack
    bsr = blocksize
    while bsr >= 8:
        result.insert(0, pack('>Q', n & 18446744073709551615))
        n = n >> 64
        bsr -= 8
    while bsr >= 4:
        result.insert(0, pack('>I', n & 4294967295))
        n = n >> 32
        bsr -= 4
    while bsr > 0:
        result.insert(0, pack('>B', n & 255))
        n = n >> 8
        bsr -= 1
    if n == 0:
        if len(result) == 0:
            bresult = b'\x00'
        else:
            bresult = b''.join(result)
    else:
        while n > 0:
            result.insert(0, pack('>Q', n & 18446744073709551615))
            n = n >> 64
        result[0] = result[0].lstrip(b'\x00')
        bresult = b''.join(result)
        if blocksize > 0:
            target_len = ((len(bresult) - 1) // blocksize + 1) * blocksize
            bresult = b'\x00' * (target_len - len(bresult)) + bresult
    return bresult