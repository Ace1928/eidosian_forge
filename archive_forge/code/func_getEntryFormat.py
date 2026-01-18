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
@staticmethod
def getEntryFormat(mapping):
    ored = 0
    for idx in mapping:
        ored |= idx
    inner = ored & 65535
    innerBits = 0
    while inner:
        innerBits += 1
        inner >>= 1
    innerBits = max(innerBits, 1)
    assert innerBits <= 16
    ored = ored >> 16 - innerBits | ored & (1 << innerBits) - 1
    if ored <= 255:
        entrySize = 1
    elif ored <= 65535:
        entrySize = 2
    elif ored <= 16777215:
        entrySize = 3
    else:
        entrySize = 4
    return entrySize - 1 << 4 | innerBits - 1