import itertools
import logging
from typing import Callable, Iterable, Optional, Mapping
from fontTools.misc.roundTools import otRound
from fontTools.ttLib import ttFont
from fontTools.ttLib.tables import _g_l_y_f
from fontTools.ttLib.tables import _h_m_t_x
from fontTools.pens.ttGlyphPen import TTGlyphPen
import pathops
def componentsOverlap(glyph: _g_l_y_f.Glyph, glyphSet: _TTGlyphMapping) -> bool:
    if not glyph.isComposite():
        raise ValueError('This method only works with TrueType composite glyphs')
    if len(glyph.components) < 2:
        return False
    component_paths = {}

    def _get_nth_component_path(index: int) -> pathops.Path:
        if index not in component_paths:
            component_paths[index] = skPathFromGlyphComponent(glyph.components[index], glyphSet)
        return component_paths[index]
    return any((pathops.op(_get_nth_component_path(i), _get_nth_component_path(j), pathops.PathOp.INTERSECTION, fix_winding=False, keep_starting_points=False) for i, j in itertools.combinations(range(len(glyph.components)), 2)))