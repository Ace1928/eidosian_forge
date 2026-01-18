from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
@staticmethod
def decompileCoord_(axisTags, data, offset):
    coord = {}
    pos = offset
    for axis in axisTags:
        coord[axis] = fi2fl(struct.unpack('>h', data[pos:pos + 2])[0], 14)
        pos += 2
    return (coord, pos)