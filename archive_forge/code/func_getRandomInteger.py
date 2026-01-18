import math
import sys
import struct
from Cryptodome import Random
from Cryptodome.Util.py3compat import iter_range
import struct
import warnings
def getRandomInteger(N, randfunc=None):
    """Return a random number at most N bits long.

    If :data:`randfunc` is omitted, then :meth:`Random.get_random_bytes` is used.

    .. deprecated:: 3.0
        This function is for internal use only and may be renamed or removed in
        the future. Use :func:`Cryptodome.Random.random.getrandbits` instead.
    """
    if randfunc is None:
        randfunc = Random.get_random_bytes
    S = randfunc(N >> 3)
    odd_bits = N % 8
    if odd_bits != 0:
        rand_bits = ord(randfunc(1)) >> 8 - odd_bits
        S = struct.pack('B', rand_bits) + S
    value = bytes_to_long(S)
    return value