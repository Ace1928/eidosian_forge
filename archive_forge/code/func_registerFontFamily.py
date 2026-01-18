import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
def registerFontFamily(family, normal=None, bold=None, italic=None, boldItalic=None):
    from reportlab.lib import fonts
    if not normal:
        normal = family
    family = family.lower()
    if not boldItalic:
        boldItalic = italic or bold or normal
    if not bold:
        bold = normal
    if not italic:
        italic = normal
    fonts.addMapping(family, 0, 0, normal)
    fonts.addMapping(family, 1, 0, bold)
    fonts.addMapping(family, 0, 1, italic)
    fonts.addMapping(family, 1, 1, boldItalic)