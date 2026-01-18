import itertools
import logging
from typing import Callable, Iterable, Optional, Mapping
from fontTools.misc.roundTools import otRound
from fontTools.ttLib import ttFont
from fontTools.ttLib.tables import _g_l_y_f
from fontTools.ttLib.tables import _h_m_t_x
from fontTools.pens.ttGlyphPen import TTGlyphPen
import pathops
def removeOverlaps(font: ttFont.TTFont, glyphNames: Optional[Iterable[str]]=None, removeHinting: bool=True, ignoreErrors=False) -> None:
    """Simplify glyphs in TTFont by merging overlapping contours.

    Overlapping components are first decomposed to simple contours, then merged.

    Currently this only works with TrueType fonts with 'glyf' table.
    Raises NotImplementedError if 'glyf' table is absent.

    Note that removing overlaps invalidates the hinting. By default we drop hinting
    from all glyphs whether or not overlaps are removed from a given one, as it would
    look weird if only some glyphs are left (un)hinted.

    Args:
        font: input TTFont object, modified in place.
        glyphNames: optional iterable of glyph names (str) to remove overlaps from.
            By default, all glyphs in the font are processed.
        removeHinting (bool): set to False to keep hinting for unmodified glyphs.
        ignoreErrors (bool): set to True to ignore errors while removing overlaps,
            thus keeping the tricky glyphs unchanged (fonttools/fonttools#2363).
    """
    try:
        glyfTable = font['glyf']
    except KeyError:
        raise NotImplementedError('removeOverlaps currently only works with TTFs')
    hmtxTable = font['hmtx']
    glyphSet = font.getGlyphSet()
    if glyphNames is None:
        glyphNames = font.getGlyphOrder()
    glyphNames = sorted(glyphNames, key=lambda name: (glyfTable[name].getCompositeMaxpValues(glyfTable).maxComponentDepth if glyfTable[name].isComposite() else 0, name))
    modified = set()
    for glyphName in glyphNames:
        try:
            if removeTTGlyphOverlaps(glyphName, glyphSet, glyfTable, hmtxTable, removeHinting):
                modified.add(glyphName)
        except RemoveOverlapsError:
            if not ignoreErrors:
                raise
            log.error("Failed to remove overlaps for '%s'", glyphName)
    log.debug('Removed overlaps for %s glyphs:\n%s', len(modified), ' '.join(modified))