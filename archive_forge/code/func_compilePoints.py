from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
@staticmethod
def compilePoints(points):
    if not points:
        return b'\x00'
    points = list(points)
    points.sort()
    numPoints = len(points)
    result = bytearray()
    if numPoints < 128:
        result.append(numPoints)
    else:
        result.append(numPoints >> 8 | 128)
        result.append(numPoints & 255)
    MAX_RUN_LENGTH = 127
    pos = 0
    lastValue = 0
    while pos < numPoints:
        runLength = 0
        headerPos = len(result)
        result.append(0)
        useByteEncoding = None
        while pos < numPoints and runLength <= MAX_RUN_LENGTH:
            curValue = points[pos]
            delta = curValue - lastValue
            if useByteEncoding is None:
                useByteEncoding = 0 <= delta <= 255
            if useByteEncoding and (delta > 255 or delta < 0):
                break
            if useByteEncoding:
                result.append(delta)
            else:
                result.append(delta >> 8)
                result.append(delta & 255)
            lastValue = curValue
            pos += 1
            runLength += 1
        if useByteEncoding:
            result[headerPos] = runLength - 1
        else:
            result[headerPos] = runLength - 1 | POINTS_ARE_WORDS
    return result