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
def setupVerticalOrigins(self, verticalOrigins, defaultVerticalOrigin=None):
    """Create a new `VORG` table. The `verticalOrigins` argument must be
        a dict, mapping glyph names to vertical origin values.

        The `defaultVerticalOrigin` argument should be the most common vertical
        origin value. If omitted, this value will be derived from the actual
        values in the `verticalOrigins` argument.
        """
    if defaultVerticalOrigin is None:
        bag = {}
        for gn in verticalOrigins:
            vorg = verticalOrigins[gn]
            if vorg not in bag:
                bag[vorg] = 1
            else:
                bag[vorg] += 1
        defaultVerticalOrigin = sorted(bag, key=lambda vorg: bag[vorg], reverse=True)[0]
    self._initTableWithValues('VORG', {}, dict(VOriginRecords={}, defaultVertOriginY=defaultVerticalOrigin))
    vorgTable = self.font['VORG']
    vorgTable.majorVersion = 1
    vorgTable.minorVersion = 0
    for gn in verticalOrigins:
        vorgTable[gn] = verticalOrigins[gn]