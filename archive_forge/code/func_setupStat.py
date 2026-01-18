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
def setupStat(self, axes, locations=None, elidedFallbackName=2):
    """Build a new 'STAT' table.

        See `fontTools.otlLib.builder.buildStatTable` for details about
        the arguments.
        """
    from .otlLib.builder import buildStatTable
    buildStatTable(self.font, axes, locations, elidedFallbackName)