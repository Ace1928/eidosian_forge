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
def setupCPAL(self, palettes, paletteTypes=None, paletteLabels=None, paletteEntryLabels=None):
    """Build new CPAL table using list of palettes.

        Optionally build CPAL v1 table using paletteTypes, paletteLabels and
        paletteEntryLabels.

        Cf. `fontTools.colorLib.builder.buildCPAL`.
        """
    from fontTools.colorLib.builder import buildCPAL
    self.font['CPAL'] = buildCPAL(palettes, paletteTypes=paletteTypes, paletteLabels=paletteLabels, paletteEntryLabels=paletteEntryLabels, nameTable=self.font.get('name'))