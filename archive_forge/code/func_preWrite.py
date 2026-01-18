import copy
from enum import IntEnum
from functools import reduce
from math import radians
import itertools
from collections import defaultdict, namedtuple
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.misc.arrayTools import quantizeRect
from fontTools.misc.roundTools import otRound
from fontTools.misc.transform import Transform, Identity
from fontTools.misc.textTools import bytesjoin, pad, safeEval
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.transformPen import TransformPen
from .otBase import (
from fontTools.feaLib.lookupDebugInfo import LookupDebugInfo, LOOKUP_DEBUG_INFO_KEY
import logging
import struct
from typing import TYPE_CHECKING, Iterator, List, Optional, Set
def preWrite(self, font):
    if not hasattr(self, 'clips'):
        self.clips = {}
    clipBoxRanges = {}
    glyphMap = font.getReverseGlyphMap()
    for glyphs, clipBox in self.groups().items():
        glyphIDs = sorted((glyphMap[glyphName] for glyphName in glyphs if glyphName in glyphMap))
        if not glyphIDs:
            continue
        last = glyphIDs[0]
        ranges = [[last]]
        for glyphID in glyphIDs[1:]:
            if glyphID != last + 1:
                ranges[-1].append(last)
                ranges.append([glyphID])
            last = glyphID
        ranges[-1].append(last)
        for start, end in ranges:
            assert (start, end) not in clipBoxRanges
            clipBoxRanges[start, end] = clipBox
    clipRecords = []
    for (start, end), clipBox in sorted(clipBoxRanges.items()):
        record = ClipRecord()
        record.StartGlyphID = start
        record.EndGlyphID = end
        record.ClipBox = clipBox
        clipRecords.append(record)
    rawTable = {'ClipCount': len(clipRecords), 'ClipRecord': clipRecords}
    return rawTable