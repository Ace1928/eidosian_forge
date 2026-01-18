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
def setGlyphOrder(self, glyphOrder):
    """Set the glyph order

        Args:
                glyphOrder ([str]): List of glyph names in order.
        """
    self.glyphOrder = glyphOrder
    if hasattr(self, '_reverseGlyphOrderDict'):
        del self._reverseGlyphOrderDict
    if self.isLoaded('glyf'):
        self['glyf'].setGlyphOrder(glyphOrder)