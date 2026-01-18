import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
def dumpFontData():
    print('Registered Encodings:')
    keys = list(_encodings.keys())
    keys.sort()
    for encName in keys:
        print('   ', encName)
    print()
    print('Registered Typefaces:')
    faces = list(_typefaces.keys())
    faces.sort()
    for faceName in faces:
        print('   ', faceName)
    print()
    print('Registered Fonts:')
    k = list(_fonts.keys())
    k.sort()
    for key in k:
        font = _fonts[key]
        print('    %s (%s/%s)' % (font.fontName, font.face.name, font.encoding.name))