import os
import zlib
import time  # noqa
import logging
import numpy as np
def int2bits(i, n=None):
    """convert int to a string of bits (0's and 1's in a string),
    pad to n elements. Convert back using int(ss,2)."""
    ii = i
    bb = BitArray()
    while ii > 0:
        bb += str(ii % 2)
        ii = ii >> 1
    bb.reverse()
    if n is not None:
        if len(bb) > n:
            raise ValueError('int2bits fail: len larger than padlength.')
        bb = str(bb).rjust(n, '0')
    return BitArray(bb)