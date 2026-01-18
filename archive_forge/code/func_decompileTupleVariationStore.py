from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
def decompileTupleVariationStore(tableTag, axisTags, tupleVariationCount, pointCount, sharedTuples, data, pos, dataPos):
    numAxes = len(axisTags)
    result = []
    if tupleVariationCount & TUPLES_SHARE_POINT_NUMBERS != 0:
        sharedPoints, dataPos = TupleVariation.decompilePoints_(pointCount, data, dataPos, tableTag)
    else:
        sharedPoints = []
    for _ in range(tupleVariationCount & TUPLE_COUNT_MASK):
        dataSize, flags = struct.unpack('>HH', data[pos:pos + 4])
        tupleSize = TupleVariation.getTupleSize_(flags, numAxes)
        tupleData = data[pos:pos + tupleSize]
        pointDeltaData = data[dataPos:dataPos + dataSize]
        result.append(decompileTupleVariation_(pointCount, sharedTuples, sharedPoints, tableTag, axisTags, tupleData, pointDeltaData))
        pos += tupleSize
        dataPos += dataSize
    return result