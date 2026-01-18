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
def calcGlyphBounds(self):
    """Calculate the bounding boxes of all glyphs in the `glyf` table.
        This is usually not called explicitly by client code.
        """
    glyphTable = self.font['glyf']
    for glyph in glyphTable.glyphs.values():
        glyph.recalcBounds(glyphTable)