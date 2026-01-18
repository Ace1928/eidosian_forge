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
def importXML(self, fileOrPath, quiet=None):
    """Import a TTX file (an XML-based text format), so as to recreate
        a font object.
        """
    if quiet is not None:
        deprecateArgument('quiet', 'configure logging instead')
    if 'maxp' in self and 'post' in self:
        self.getGlyphOrder()
    from fontTools.misc import xmlReader
    reader = xmlReader.XMLReader(fileOrPath, self)
    reader.read()