import math
import sys
import struct
from Cryptodome import Random
from Cryptodome.Util.py3compat import iter_range
import struct
import warnings
def getRandomNBitInteger(N, randfunc=None):
    """Return a random number with exactly N-bits,
    i.e. a random number between 2**(N-1) and (2**N)-1.

    If :data:`randfunc` is omitted, then :meth:`Random.get_random_bytes` is used.

    .. deprecated:: 3.0
        This function is for internal use only and may be renamed or removed in
        the future.
    """
    value = getRandomInteger(N - 1, randfunc)
    value |= 2 ** (N - 1)
    assert size(value) >= N
    return value