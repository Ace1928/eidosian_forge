import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
def findFontAndRegister(fontName):
    """search for and register a font given its name"""
    fontName = str(fontName)
    assert type(fontName) is str, 'fontName=%s is not required type str' % ascii(fontName)
    face = getTypeFace(fontName)
    if face.requiredEncoding:
        font = Font(fontName, fontName, face.requiredEncoding)
    else:
        font = Font(fontName, fontName, defaultEncoding)
    registerFont(font)
    return font