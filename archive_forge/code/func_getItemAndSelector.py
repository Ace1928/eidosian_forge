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
def getItemAndSelector(self, name):
    if self.charStringsAreIndexed:
        index = self.charStrings[name]
        return self.charStringsIndex.getItemAndSelector(index)
    else:
        if hasattr(self, 'fdArray'):
            if hasattr(self, 'fdSelect'):
                sel = self.charStrings[name].fdSelectIndex
            else:
                sel = 0
        else:
            sel = None
        return (self.charStrings[name], sel)