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
def setupMetrics(self, tableTag, metrics):
    """See `setupHorizontalMetrics()` and `setupVerticalMetrics()`."""
    assert tableTag in ('hmtx', 'vmtx')
    mtxTable = self.font[tableTag] = newTable(tableTag)
    roundedMetrics = {}
    for gn in metrics:
        w, lsb = metrics[gn]
        roundedMetrics[gn] = (int(round(w)), int(round(lsb)))
    mtxTable.metrics = roundedMetrics