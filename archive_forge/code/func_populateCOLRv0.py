import collections
import copy
import enum
from functools import partial
from math import ceil, log
from typing import (
from fontTools.misc.arrayTools import intRect
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import C_O_L_R_
from fontTools.ttLib.tables import C_P_A_L_
from fontTools.ttLib.tables import _n_a_m_e
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otTables import ExtendMode, CompositeMode
from .errors import ColorLibError
from .geometry import round_start_circle_stable_containment
from .table_builder import BuildCallback, TableBuilder
def populateCOLRv0(table: ot.COLR, colorGlyphsV0: _ColorGlyphsV0Dict, glyphMap: Optional[Mapping[str, int]]=None):
    """Build v0 color layers and add to existing COLR table.

    Args:
        table: a raw ``otTables.COLR()`` object (not ttLib's ``table_C_O_L_R_``).
        colorGlyphsV0: map of base glyph names to lists of (layer glyph names,
            color palette index) tuples. Can be empty.
        glyphMap: a map from glyph names to glyph indices, as returned from
            ``TTFont.getReverseGlyphMap()``, to optionally sort base records by GID.
    """
    if glyphMap is not None:
        colorGlyphItems = sorted(colorGlyphsV0.items(), key=lambda item: glyphMap[item[0]])
    else:
        colorGlyphItems = colorGlyphsV0.items()
    baseGlyphRecords = []
    layerRecords = []
    for baseGlyph, layers in colorGlyphItems:
        baseRec = ot.BaseGlyphRecord()
        baseRec.BaseGlyph = baseGlyph
        baseRec.FirstLayerIndex = len(layerRecords)
        baseRec.NumLayers = len(layers)
        baseGlyphRecords.append(baseRec)
        for layerGlyph, paletteIndex in layers:
            layerRec = ot.LayerRecord()
            layerRec.LayerGlyph = layerGlyph
            layerRec.PaletteIndex = paletteIndex
            layerRecords.append(layerRec)
    table.BaseGlyphRecordArray = table.LayerRecordArray = None
    if baseGlyphRecords:
        table.BaseGlyphRecordArray = ot.BaseGlyphRecordArray()
        table.BaseGlyphRecordArray.BaseGlyphRecord = baseGlyphRecords
    if layerRecords:
        table.LayerRecordArray = ot.LayerRecordArray()
        table.LayerRecordArray.LayerRecord = layerRecords
    table.BaseGlyphRecordCount = len(baseGlyphRecords)
    table.LayerRecordCount = len(layerRecords)