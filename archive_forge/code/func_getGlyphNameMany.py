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
def getGlyphNameMany(self, lst):
    """Converts a list of glyph IDs into a list of glyph names."""
    glyphOrder = self.getGlyphOrder()
    cnt = len(glyphOrder)
    return [glyphOrder[gid] if gid < cnt else 'glyph%.5d' % gid for gid in lst]