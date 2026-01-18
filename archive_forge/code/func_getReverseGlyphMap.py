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
def getReverseGlyphMap(self, rebuild=False):
    """Returns a mapping of glyph names to glyph IDs."""
    if rebuild or not hasattr(self, '_reverseGlyphOrderDict'):
        self._buildReverseGlyphOrderDict()
    return self._reverseGlyphOrderDict