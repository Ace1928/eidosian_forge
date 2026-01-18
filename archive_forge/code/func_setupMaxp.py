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
def setupMaxp(self):
    """Create a new `maxp` table. This is called implicitly by FontBuilder
        itself and is usually not called by client code.
        """
    if self.isTTF:
        defaults = _maxpDefaultsTTF
    else:
        defaults = _maxpDefaultsOTF
    self._initTableWithValues('maxp', defaults, {})