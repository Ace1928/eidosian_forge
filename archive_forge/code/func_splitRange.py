from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
def splitRange(startCode, endCode, cmap):
    if startCode == endCode:
        return ([], [endCode])
    lastID = cmap[startCode]
    lastCode = startCode
    inOrder = None
    orderedBegin = None
    subRanges = []
    for code in range(startCode + 1, endCode + 1):
        glyphID = cmap[code]
        if glyphID - 1 == lastID:
            if inOrder is None or not inOrder:
                inOrder = 1
                orderedBegin = lastCode
        elif inOrder:
            inOrder = 0
            subRanges.append((orderedBegin, lastCode))
            orderedBegin = None
        lastID = glyphID
        lastCode = code
    if inOrder:
        subRanges.append((orderedBegin, lastCode))
    assert lastCode == endCode
    newRanges = []
    for b, e in subRanges:
        if b == startCode and e == endCode:
            break
        if b == startCode or e == endCode:
            threshold = 4
        else:
            threshold = 8
        if e - b + 1 > threshold:
            newRanges.append((b, e))
    subRanges = newRanges
    if not subRanges:
        return ([], [endCode])
    if subRanges[0][0] != startCode:
        subRanges.insert(0, (startCode, subRanges[0][0] - 1))
    if subRanges[-1][1] != endCode:
        subRanges.append((subRanges[-1][1] + 1, endCode))
    i = 1
    while i < len(subRanges):
        if subRanges[i - 1][1] + 1 != subRanges[i][0]:
            subRanges.insert(i, (subRanges[i - 1][1] + 1, subRanges[i][0] - 1))
            i = i + 1
        i = i + 1
    start = []
    end = []
    for b, e in subRanges:
        start.append(b)
        end.append(e)
    start.pop(0)
    assert len(start) + 1 == len(end)
    return (start, end)