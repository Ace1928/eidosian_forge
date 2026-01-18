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
def computeClipBox(self, colr: COLR, glyphSet: '_TTGlyphSet', quantization: int=1) -> Optional[ClipBox]:
    pen = ControlBoundsPen(glyphSet)
    for path in dfs_base_table(self, iter_subtables_fn=lambda paint: paint.iterPaintSubTables(colr)):
        paint = path[-1].value
        if paint.Format == PaintFormat.PaintGlyph:
            transformation = reduce(Transform.transform, (st.value.getTransform() for st in path), Identity)
            glyphSet[paint.Glyph].draw(TransformPen(pen, transformation))
    if pen.bounds is None:
        return None
    cb = ClipBox()
    cb.Format = int(ClipBoxFormat.Static)
    cb.xMin, cb.yMin, cb.xMax, cb.yMax = quantizeRect(pen.bounds, quantization)
    return cb