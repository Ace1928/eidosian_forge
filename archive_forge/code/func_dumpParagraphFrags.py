from string import whitespace
from operator import truth
from unicodedata import category
from reportlab.pdfbase.pdfmetrics import stringWidth, getAscentDescent
from reportlab.platypus.paraparser import ParaParser, _PCT, _num as _parser_num, _re_us_value
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import Color
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.geomutils import normalizeTRBL
from reportlab.lib.textsplit import wordSplit, ALL_CANNOT_START
from reportlab.lib.styles import ParagraphStyle
from copy import deepcopy
from reportlab.lib.abag import ABag
from reportlab.rl_config import decimalSymbol, _FUZZ, paraFontSizeHeightOffset,\
from reportlab.lib.utils import _className, isBytes, isStr
from reportlab.lib.rl_accel import sameFrag
import re
from types import MethodType
def dumpParagraphFrags(P):
    print('dumpParagraphFrags(<Paragraph @ %d>) minWidth() = %.2f' % (id(P), P.minWidth()))
    frags = P.frags
    n = len(frags)
    for l in range(n):
        print("frag%d: '%s' %s" % (l, frags[l].text, ' '.join(['%s=%s' % (k, getattr(frags[l], k)) for k in frags[l].__dict__ if k != text])))
    outw = sys.stdout.write
    l = 0
    cum = 0
    for W in _getFragWords(frags, 360):
        cum += W[0]
        outw('fragword%d: cum=%3d size=%d' % (l, cum, W[0]))
        for w in W[1:]:
            outw(' (%s)' % fragDump(w))
        print()
        l += 1