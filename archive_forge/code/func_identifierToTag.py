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
def identifierToTag(ident):
    """the opposite of tagToIdentifier()"""
    if ident == 'GlyphOrder':
        return ident
    if len(ident) % 2 and ident[0] == '_':
        ident = ident[1:]
    assert not len(ident) % 2
    tag = ''
    for i in range(0, len(ident), 2):
        if ident[i] == '_':
            tag = tag + ident[i + 1]
        elif ident[i + 1] == '_':
            tag = tag + ident[i]
        else:
            tag = tag + chr(int(ident[i:i + 2], 16))
    tag = tag + (4 - len(tag)) * ' '
    return Tag(tag)