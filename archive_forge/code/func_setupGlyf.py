from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from .ttLib import TTFont, newTable
from .ttLib.tables._c_m_a_p import cmap_classes
from .ttLib.tables._g_l_y_f import flagCubic
from .ttLib.tables.O_S_2f_2 import Panose
from .misc.timeTools import timestampNow
import struct
from collections import OrderedDict
def setupGlyf(self, glyphs, calcGlyphBounds=True, validateGlyphFormat=True):
    """Create the `glyf` table from a dict, that maps glyph names
        to `fontTools.ttLib.tables._g_l_y_f.Glyph` objects, for example
        as made by `fontTools.pens.ttGlyphPen.TTGlyphPen`.

        If `calcGlyphBounds` is True, the bounds of all glyphs will be
        calculated. Only pass False if your glyph objects already have
        their bounding box values set.

        If `validateGlyphFormat` is True, raise ValueError if any of the glyphs contains
        cubic curves or is a variable composite but head.glyphDataFormat=0.
        Set it to False to skip the check if you know in advance all the glyphs are
        compatible with the specified glyphDataFormat.
        """
    assert self.isTTF
    if validateGlyphFormat and self.font['head'].glyphDataFormat == 0:
        for name, g in glyphs.items():
            if g.isVarComposite():
                raise ValueError(f'Glyph {name!r} is a variable composite, but glyphDataFormat=0')
            elif g.numberOfContours > 0 and any((f & flagCubic for f in g.flags)):
                raise ValueError(f'Glyph {name!r} has cubic Bezier outlines, but glyphDataFormat=0; either convert to quadratics with cu2qu or set glyphDataFormat=1.')
    self.font['loca'] = newTable('loca')
    self.font['glyf'] = newTable('glyf')
    self.font['glyf'].glyphs = glyphs
    if hasattr(self.font, 'glyphOrder'):
        self.font['glyf'].glyphOrder = self.font.glyphOrder
    if calcGlyphBounds:
        self.calcGlyphBounds()