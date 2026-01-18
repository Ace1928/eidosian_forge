import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
def registerFont(font):
    """Registers a font, including setting up info for accelerated stringWidth"""
    fontName = font.fontName
    if font._dynamicFont:
        faceName = font.face.name
        if fontName not in _fonts:
            if faceName in _dynFaceNames:
                ofont = _dynFaceNames[faceName]
                if not ofont._dynamicFont:
                    raise ValueError('Attempt to register fonts %r %r for face %r' % (ofont, font, faceName))
                else:
                    _fonts[fontName] = ofont
            else:
                _dynFaceNames[faceName] = _fonts[fontName] = font
    else:
        _fonts[fontName] = font
    if font._multiByte:
        registerFontFamily(font.fontName)