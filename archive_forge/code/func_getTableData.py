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
def getTableData(self, tag):
    """Returns the binary representation of a table.

        If the table is currently loaded and in memory, the data is compiled to
        binary and returned; if it is not currently loaded, the binary data is
        read from the font file and returned.
        """
    tag = Tag(tag)
    if self.isLoaded(tag):
        log.debug("Compiling '%s' table", tag)
        return self.tables[tag].compile(self)
    elif self.reader and tag in self.reader:
        log.debug("Reading '%s' table from disk", tag)
        return self.reader[tag]
    else:
        raise KeyError(tag)