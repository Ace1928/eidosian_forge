from fontTools.config import Config
from fontTools.misc import xmlWriter
from fontTools.misc.configTools import AbstractConfig
from fontTools.misc.textTools import Tag, byteord, tostr
from fontTools.misc.loggingTools import deprecateArgument
from fontTools.ttLib import TTLibError
from fontTools.ttLib.ttGlyphSet import _TTGlyph, _TTGlyphSetCFF, _TTGlyphSetGlyf
from fontTools.ttLib.sfnt import SFNTReader, SFNTWriter
from io import BytesIO, StringIO, UnsupportedOperation
import os
import logging
import traceback
def getSearchRange(n, itemSize=16):
    """Calculate searchRange, entrySelector, rangeShift."""
    exponent = maxPowerOfTwo(n)
    searchRange = 2 ** exponent * itemSize
    entrySelector = exponent
    rangeShift = max(0, n * itemSize - searchRange)
    return (searchRange, entrySelector, rangeShift)