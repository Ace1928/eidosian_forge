from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
from fontTools.ttLib.sfnt import (
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging
def reconstructTable(self, tag):
    """Reconstruct table named 'tag' from transformed data."""
    entry = self.tables[Tag(tag)]
    rawData = entry.loadData(self.transformBuffer)
    if tag == 'glyf':
        padding = self.padding if hasattr(self, 'padding') else None
        data = self._reconstructGlyf(rawData, padding)
    elif tag == 'loca':
        data = self._reconstructLoca()
    elif tag == 'hmtx':
        data = self._reconstructHmtx(rawData)
    else:
        raise TTLibError("transform for table '%s' is unknown" % tag)
    return data