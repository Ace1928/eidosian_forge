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
def splitPairPos(oldSubTable, newSubTable, overflowRecord):
    st = oldSubTable
    ok = False
    newSubTable.Format = oldSubTable.Format
    if oldSubTable.Format == 1 and len(oldSubTable.PairSet) > 1:
        for name in ('ValueFormat1', 'ValueFormat2'):
            setattr(newSubTable, name, getattr(oldSubTable, name))
        newSubTable.Coverage = oldSubTable.Coverage.__class__()
        coverage = oldSubTable.Coverage.glyphs
        records = oldSubTable.PairSet
        oldCount = len(oldSubTable.PairSet) // 2
        oldSubTable.Coverage.glyphs = coverage[:oldCount]
        oldSubTable.PairSet = records[:oldCount]
        newSubTable.Coverage.glyphs = coverage[oldCount:]
        newSubTable.PairSet = records[oldCount:]
        oldSubTable.PairSetCount = len(oldSubTable.PairSet)
        newSubTable.PairSetCount = len(newSubTable.PairSet)
        ok = True
    elif oldSubTable.Format == 2 and len(oldSubTable.Class1Record) > 1:
        if not hasattr(oldSubTable, 'Class2Count'):
            oldSubTable.Class2Count = len(oldSubTable.Class1Record[0].Class2Record)
        for name in ('Class2Count', 'ClassDef2', 'ValueFormat1', 'ValueFormat2'):
            setattr(newSubTable, name, getattr(oldSubTable, name))
        oldSubTable.DontShare = True
        newSubTable.Coverage = oldSubTable.Coverage.__class__()
        newSubTable.ClassDef1 = oldSubTable.ClassDef1.__class__()
        coverage = oldSubTable.Coverage.glyphs
        classDefs = oldSubTable.ClassDef1.classDefs
        records = oldSubTable.Class1Record
        oldCount = len(oldSubTable.Class1Record) // 2
        newGlyphs = set((k for k, v in classDefs.items() if v >= oldCount))
        oldSubTable.Coverage.glyphs = [g for g in coverage if g not in newGlyphs]
        oldSubTable.ClassDef1.classDefs = {k: v for k, v in classDefs.items() if v < oldCount}
        oldSubTable.Class1Record = records[:oldCount]
        newSubTable.Coverage.glyphs = [g for g in coverage if g in newGlyphs]
        newSubTable.ClassDef1.classDefs = {k: v - oldCount for k, v in classDefs.items() if v > oldCount}
        newSubTable.Class1Record = records[oldCount:]
        oldSubTable.Class1Count = len(oldSubTable.Class1Record)
        newSubTable.Class1Count = len(newSubTable.Class1Record)
        ok = True
    return ok