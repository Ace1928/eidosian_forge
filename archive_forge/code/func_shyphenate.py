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
def shyphenate(self, newWidth, maxWidth):
    ww = self[0]
    self._fsww = 2147483647
    if ww == 0:
        return []
    possible = None
    exceeded = False
    baseWidth = baseWidth0 = newWidth - ww
    fsww = None
    for i, (f, t) in enumerate(self[1:]):
        sW = lambda s: stringWidth(s, f.fontName, f.fontSize)
        if isinstance(t, _SHYIndexedStr):
            shyLen = sW(u'-')
            bw = baseWidth + shyLen
            for j, x in enumerate(t._shyIndices):
                left, right = (t[:x], t[x:])
                leftw = bw + sW(left)
                if fsww is None:
                    fsww = leftw
                exceeded = leftw > maxWidth
                if exceeded:
                    break
                possible = (i, j, x, leftw, left, right, shyLen)
            baseWidth += sW(t)
        else:
            baseWidth += sW(t)
            exceeded = baseWidth > maxWidth
        if exceeded and fsww is not None:
            break
    self._fsww = fsww - baseWidth0 if fsww is not None else 2147483647
    if not possible:
        return []
    i, j, x, leftw, left, right, shyLen = possible
    i1 = i + 1
    f, t = self[i1]
    X = t._shyIndices
    lefts = _SHYIndexedStr(left + u'-', X[:j + 1])
    L = self[:i1] + [(f, lefts)]
    L[0] = leftw - baseWidth0
    R = [ww - L[0] + shyLen] + ([] if not right else [(f, _SHYIndexedStr(right, [_ - x for _ in X[j + 1:]]))]) + self[i1 + 1:]
    return (_SplitFragSHY(L), _SHYWordHS(R))