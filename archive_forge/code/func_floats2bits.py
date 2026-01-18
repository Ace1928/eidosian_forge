import os
import zlib
import time  # noqa
import logging
import numpy as np
def floats2bits(arr):
    """Given a few (signed) numbers, convert them to bits,
    stored as FB (float bit values). We always use 16.16.
    Negative numbers are not (yet) possible, because I don't
    know how the're implemented (ambiguity).
    """
    bits = int2bits(31, 5)
    for i in arr:
        if i < 0:
            raise ValueError('Dit not implement negative floats!')
        i1 = int(i)
        i2 = i - i1
        bits += int2bits(i1, 15)
        bits += int2bits(i2 * 2 ** 16, 16)
    return bits