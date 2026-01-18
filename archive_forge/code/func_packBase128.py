from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
from fontTools.ttLib.sfnt import (
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging
def packBase128(n):
    """Encode unsigned integer in range 0 to 2**32-1 (inclusive) to a string of
    bytes using UIntBase128 variable-length encoding. Produce the shortest possible
    encoding.

    >>> packBase128(63) == b"\\x3f"
    True
    >>> packBase128(2**32-1) == b'\\x8f\\xff\\xff\\xff\\x7f'
    True
    """
    if n < 0 or n >= 2 ** 32:
        raise TTLibError('UIntBase128 format requires 0 <= integer <= 2**32-1')
    data = b''
    size = base128Size(n)
    for i in range(size):
        b = n >> 7 * (size - i - 1) & 127
        if i < size - 1:
            b |= 128
        data += struct.pack('B', b)
    return data