from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
@staticmethod
def decompilePoints_(numPoints, data, offset, tableTag):
    """(numPoints, data, offset, tableTag) --> ([point1, point2, ...], newOffset)"""
    assert tableTag in ('cvar', 'gvar')
    pos = offset
    numPointsInData = data[pos]
    pos += 1
    if numPointsInData & POINTS_ARE_WORDS != 0:
        numPointsInData = (numPointsInData & POINT_RUN_COUNT_MASK) << 8 | data[pos]
        pos += 1
    if numPointsInData == 0:
        return (range(numPoints), pos)
    result = []
    while len(result) < numPointsInData:
        runHeader = data[pos]
        pos += 1
        numPointsInRun = (runHeader & POINT_RUN_COUNT_MASK) + 1
        point = 0
        if runHeader & POINTS_ARE_WORDS != 0:
            points = array.array('H')
            pointsSize = numPointsInRun * 2
        else:
            points = array.array('B')
            pointsSize = numPointsInRun
        points.frombytes(data[pos:pos + pointsSize])
        if sys.byteorder != 'big':
            points.byteswap()
        assert len(points) == numPointsInRun
        pos += pointsSize
        result.extend(points)
    absolute = []
    current = 0
    for delta in result:
        current += delta
        absolute.append(current)
    result = absolute
    del absolute
    badPoints = {str(p) for p in result if p < 0 or p >= numPoints}
    if badPoints:
        log.warning("point %s out of range in '%s' table" % (','.join(sorted(badPoints)), tableTag))
    return (result, pos)