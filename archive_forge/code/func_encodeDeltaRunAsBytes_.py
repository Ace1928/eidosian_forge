from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
@staticmethod
def encodeDeltaRunAsBytes_(deltas, offset, bytearr):
    pos = offset
    numDeltas = len(deltas)
    while pos < numDeltas:
        value = deltas[pos]
        if not -128 <= value <= 127:
            break
        if value == 0 and pos + 1 < numDeltas and (deltas[pos + 1] == 0):
            break
        pos += 1
    runLength = pos - offset
    while runLength >= 64:
        bytearr.append(63)
        bytearr.extend(array.array('b', deltas[offset:offset + 64]))
        offset += 64
        runLength -= 64
    if runLength:
        bytearr.append(runLength - 1)
        bytearr.extend(array.array('b', deltas[offset:pos]))
    return pos