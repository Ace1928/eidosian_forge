from fontTools.misc import sstruct
from fontTools.misc import psCharStrings
from fontTools.misc.arrayTools import unionRect, intRect
from fontTools.misc.textTools import (
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.ttLib.tables.otBase import OTTableReader
from fontTools.ttLib.tables import otTables as ot
from io import BytesIO
import struct
import logging
import re
def recalcFontBBox(self):
    fontBBox = None
    for charString in self.CharStrings.values():
        bounds = charString.calcBounds(self.CharStrings)
        if bounds is not None:
            if fontBBox is not None:
                fontBBox = unionRect(fontBBox, bounds)
            else:
                fontBBox = bounds
    if fontBBox is None:
        self.FontBBox = self.defaults['FontBBox'][:]
    else:
        self.FontBBox = list(intRect(fontBBox))