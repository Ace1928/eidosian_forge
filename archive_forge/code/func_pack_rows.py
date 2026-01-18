import collections
import io   # For io.BytesIO
import itertools
import math
import operator
import re
import struct
import sys
import warnings
import zlib
from array import array
fromarray = from_array
def pack_rows(rows, bitdepth):
    """Yield packed rows that are a byte array.
    Each byte is packed with the values from several pixels.
    """
    assert bitdepth < 8
    assert 8 % bitdepth == 0
    spb = int(8 / bitdepth)

    def make_byte(block):
        """Take a block of (2, 4, or 8) values,
        and pack them into a single byte.
        """
        res = 0
        for v in block:
            res = (res << bitdepth) + v
        return res
    for row in rows:
        a = bytearray(row)
        n = float(len(a))
        extra = math.ceil(n / spb) * spb - n
        a.extend([0] * int(extra))
        blocks = group(a, spb)
        yield bytearray((make_byte(block) for block in blocks))