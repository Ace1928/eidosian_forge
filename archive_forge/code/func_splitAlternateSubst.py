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
def splitAlternateSubst(oldSubTable, newSubTable, overflowRecord):
    ok = 1
    if hasattr(oldSubTable, 'sortCoverageLast'):
        newSubTable.sortCoverageLast = oldSubTable.sortCoverageLast
    oldAlts = sorted(oldSubTable.alternates.items())
    oldLen = len(oldAlts)
    if overflowRecord.itemName in ['Coverage', 'RangeRecord']:
        newLen = oldLen // 2
    elif overflowRecord.itemName == 'AlternateSet':
        newLen = overflowRecord.itemIndex - 1
    newSubTable.alternates = {}
    for i in range(newLen, oldLen):
        item = oldAlts[i]
        key = item[0]
        newSubTable.alternates[key] = item[1]
        del oldSubTable.alternates[key]
    return ok