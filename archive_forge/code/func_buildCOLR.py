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
def buildCOLR(colorGlyphs: _ColorGlyphsDict, version: Optional[int]=None, *, glyphMap: Optional[Mapping[str, int]]=None, varStore: Optional[ot.VarStore]=None, varIndexMap: Optional[ot.DeltaSetIndexMap]=None, clipBoxes: Optional[Dict[str, _ClipBoxInput]]=None, allowLayerReuse: bool=True) -> C_O_L_R_.table_C_O_L_R_:
    """Build COLR table from color layers mapping.

    Args:

        colorGlyphs: map of base glyph name to, either list of (layer glyph name,
            color palette index) tuples for COLRv0; or a single ``Paint`` (dict) or
            list of ``Paint`` for COLRv1.
        version: the version of COLR table. If None, the version is determined
            by the presence of COLRv1 paints or variation data (varStore), which
            require version 1; otherwise, if all base glyphs use only simple color
            layers, version 0 is used.
        glyphMap: a map from glyph names to glyph indices, as returned from
            TTFont.getReverseGlyphMap(), to optionally sort base records by GID.
        varStore: Optional ItemVarationStore for deltas associated with v1 layer.
        varIndexMap: Optional DeltaSetIndexMap for deltas associated with v1 layer.
        clipBoxes: Optional map of base glyph name to clip box 4- or 5-tuples:
            (xMin, yMin, xMax, yMax) or (xMin, yMin, xMax, yMax, varIndexBase).

    Returns:
        A new COLR table.
    """
    self = C_O_L_R_.table_C_O_L_R_()
    if varStore is not None and version == 0:
        raise ValueError("Can't add VarStore to COLRv0")
    if version in (None, 0) and (not varStore):
        colorGlyphsV0, colorGlyphsV1 = _split_color_glyphs_by_version(colorGlyphs)
        if version == 0 and colorGlyphsV1:
            raise ValueError("Can't encode COLRv1 glyphs in COLRv0")
    else:
        colorGlyphsV0, colorGlyphsV1 = ({}, colorGlyphs)
    colr = ot.COLR()
    populateCOLRv0(colr, colorGlyphsV0, glyphMap)
    colr.LayerList, colr.BaseGlyphList = buildColrV1(colorGlyphsV1, glyphMap, allowLayerReuse=allowLayerReuse)
    if version is None:
        version = 1 if varStore or colorGlyphsV1 else 0
    elif version not in (0, 1):
        raise NotImplementedError(version)
    self.version = colr.Version = version
    if version == 0:
        self.ColorLayers = self._decompileColorLayersV0(colr)
    else:
        colr.ClipList = buildClipList(clipBoxes) if clipBoxes else None
        colr.VarIndexMap = varIndexMap
        colr.VarStore = varStore
        self.table = colr
    return self