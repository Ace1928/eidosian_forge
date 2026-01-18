from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
def compileCoord(self, axisTags):
    result = []
    axes = self.axes
    for axis in axisTags:
        triple = axes.get(axis)
        if triple is None:
            result.append(b'\x00\x00')
        else:
            result.append(struct.pack('>h', fl2fi(triple[1], 14)))
    return b''.join(result)