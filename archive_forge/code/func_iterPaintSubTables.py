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
def iterPaintSubTables(self, colr: COLR) -> Iterator[BaseTable.SubTableEntry]:
    if self.Format == PaintFormat.PaintColrLayers:
        layers = []
        if colr.LayerList is not None:
            layers = colr.LayerList.Paint
        yield from (BaseTable.SubTableEntry(name='Layers', value=v, index=i) for i, v in enumerate(layers[self.FirstLayerIndex:self.FirstLayerIndex + self.NumLayers]))
        return
    if self.Format == PaintFormat.PaintColrGlyph:
        for record in colr.BaseGlyphList.BaseGlyphPaintRecord:
            if record.BaseGlyph == self.Glyph:
                yield BaseTable.SubTableEntry(name='BaseGlyph', value=record.Paint)
                return
        else:
            raise KeyError(f'{self.Glyph!r} not in colr.BaseGlyphList')
    for conv in self.getConverters():
        if conv.tableClass is not None and issubclass(conv.tableClass, type(self)):
            value = getattr(self, conv.name)
            yield BaseTable.SubTableEntry(name=conv.name, value=value)