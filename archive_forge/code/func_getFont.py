import sys, os, pickle
from hashlib import md5
from xml.sax.saxutils import quoteattr
from time import process_time as clock
from reportlab.lib.utils import asBytes, asNative as _asNative
from reportlab.lib.utils import rl_isdir, rl_isfile, rl_listdir, rl_getmtime
def getFont(self, familyName, bold=False, italic=False):
    """Try to find a font matching the spec"""
    for font in self._fonts:
        if font.familyName == familyName:
            if font.isBold == bold:
                if font.isItalic == italic:
                    return font
    raise KeyError('Cannot find font %s with bold=%s, italic=%s' % (familyName, bold, italic))