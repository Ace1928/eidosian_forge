from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
def buildReversed(self):
    """Builds a reverse mapping dictionary

        Iterates over all Unicode cmap tables and returns a dictionary mapping
        glyphs to sets of codepoints, such as::

                {
                        'one': {0x31}
                        'A': {0x41,0x391}
                }

        The values are sets of Unicode codepoints because
        some fonts map different codepoints to the same glyph.
        For example, ``U+0041 LATIN CAPITAL LETTER A`` and ``U+0391
        GREEK CAPITAL LETTER ALPHA`` are sometimes the same glyph.
        """
    result = {}
    for subtable in self.tables:
        if subtable.isUnicode():
            for codepoint, name in subtable.cmap.items():
                result.setdefault(name, set()).add(codepoint)
    return result