from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
def decompileTupleVariation_(pointCount, sharedTuples, sharedPoints, tableTag, axisTags, data, tupleData):
    assert tableTag in ('cvar', 'gvar'), tableTag
    flags = struct.unpack('>H', data[2:4])[0]
    pos = 4
    if flags & EMBEDDED_PEAK_TUPLE == 0:
        peak = sharedTuples[flags & TUPLE_INDEX_MASK]
    else:
        peak, pos = TupleVariation.decompileCoord_(axisTags, data, pos)
    if flags & INTERMEDIATE_REGION != 0:
        start, pos = TupleVariation.decompileCoord_(axisTags, data, pos)
        end, pos = TupleVariation.decompileCoord_(axisTags, data, pos)
    else:
        start, end = inferRegion_(peak)
    axes = {}
    for axis in axisTags:
        region = (start[axis], peak[axis], end[axis])
        if region != (0.0, 0.0, 0.0):
            axes[axis] = region
    pos = 0
    if flags & PRIVATE_POINT_NUMBERS != 0:
        points, pos = TupleVariation.decompilePoints_(pointCount, tupleData, pos, tableTag)
    else:
        points = sharedPoints
    deltas = [None] * pointCount
    if tableTag == 'cvar':
        deltas_cvt, pos = TupleVariation.decompileDeltas_(len(points), tupleData, pos)
        for p, delta in zip(points, deltas_cvt):
            if 0 <= p < pointCount:
                deltas[p] = delta
    elif tableTag == 'gvar':
        deltas_x, pos = TupleVariation.decompileDeltas_(len(points), tupleData, pos)
        deltas_y, pos = TupleVariation.decompileDeltas_(len(points), tupleData, pos)
        for p, x, y in zip(points, deltas_x, deltas_y):
            if 0 <= p < pointCount:
                deltas[p] = (x, y)
    return TupleVariation(axes, deltas)