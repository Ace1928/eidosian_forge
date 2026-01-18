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
def ensureDecompiled(self, recurse=None):
    """Decompile all the tables, even if a TTFont was opened in 'lazy' mode."""
    for tag in self.keys():
        table = self[tag]
        if recurse is None:
            recurse = self.lazy is not False
        if recurse and hasattr(table, 'ensureDecompiled'):
            table.ensureDecompiled(recurse=recurse)
    self.lazy = False