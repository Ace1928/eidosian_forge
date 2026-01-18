import os
import copy
import enum
from operator import ior
import logging
from fontTools.colorLib.builder import MAX_PAINT_COLR_LAYER_COUNT, LayerReuseCache
from fontTools.misc import classifyTools
from fontTools.misc.roundTools import otRound
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables import otBase as otBase
from fontTools.ttLib.tables.otConverters import BaseFixedValue
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.varLib import builder, models, varStore
from fontTools.varLib.models import nonNone, allNone, allEqual, allEqualTo, subList
from fontTools.varLib.varStore import VarStoreInstancer
from functools import reduce
from fontTools.otlLib.builder import buildSinglePos
from fontTools.otlLib.optimize.gpos import (
from .errors import (
@staticmethod
def expandPaintColrLayers(colr):
    """Rebuild LayerList without PaintColrLayers reuse.

        Each base paint graph is fully DFS-traversed (with exception of PaintColrGlyph
        which are irrelevant for this); any layers referenced via PaintColrLayers are
        collected into a new LayerList and duplicated when reuse is detected, to ensure
        that all paints are distinct objects at the end of the process.
        PaintColrLayers's FirstLayerIndex/NumLayers are updated so that no overlap
        is left. Also, any consecutively nested PaintColrLayers are flattened.
        The COLR table's LayerList is replaced with the new unique layers.
        A side effect is also that any layer from the old LayerList which is not
        referenced by any PaintColrLayers is dropped.
        """
    if not colr.LayerList:
        return
    uniqueLayerIDs = set()
    newLayerList = []
    for rec in colr.BaseGlyphList.BaseGlyphPaintRecord:
        frontier = [rec.Paint]
        while frontier:
            paint = frontier.pop()
            if paint.Format == ot.PaintFormat.PaintColrGlyph:
                continue
            elif paint.Format == ot.PaintFormat.PaintColrLayers:
                children = list(_flatten_layers(paint, colr))
                first_layer_index = len(newLayerList)
                for layer in children:
                    if id(layer) in uniqueLayerIDs:
                        layer = copy.deepcopy(layer)
                        assert id(layer) not in uniqueLayerIDs
                    newLayerList.append(layer)
                    uniqueLayerIDs.add(id(layer))
                paint.FirstLayerIndex = first_layer_index
                paint.NumLayers = len(children)
            else:
                children = paint.getChildren(colr)
            frontier.extend(reversed(children))
    assert len(newLayerList) == len(uniqueLayerIDs)
    colr.LayerList.Paint = newLayerList
    colr.LayerList.LayerCount = len(newLayerList)