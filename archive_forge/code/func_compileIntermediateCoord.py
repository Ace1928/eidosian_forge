from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
def compileIntermediateCoord(self, axisTags):
    needed = False
    for axis in axisTags:
        minValue, value, maxValue = self.axes.get(axis, (0.0, 0.0, 0.0))
        defaultMinValue = min(value, 0.0)
        defaultMaxValue = max(value, 0.0)
        if minValue != defaultMinValue or maxValue != defaultMaxValue:
            needed = True
            break
    if not needed:
        return None
    minCoords = []
    maxCoords = []
    for axis in axisTags:
        minValue, value, maxValue = self.axes.get(axis, (0.0, 0.0, 0.0))
        minCoords.append(struct.pack('>h', fl2fi(minValue, 14)))
        maxCoords.append(struct.pack('>h', fl2fi(maxValue, 14)))
    return b''.join(minCoords + maxCoords)