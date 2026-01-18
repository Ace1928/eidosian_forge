import re
import sys
import copy
import unicodedata
import reportlab.lib.sequencer
from reportlab.lib.abag import ABag
from reportlab.lib.utils import ImageReader, annotateException, encode_label, asUnicode
from reportlab.lib.colors import toColor, black
from reportlab.lib.fonts import tt2ps, ps2tt
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import inch,mm,cm,pica
from reportlab.rl_config import platypus_link_underline
from html.parser import HTMLParser
from html.entities import name2codepoint
def start_span(self, attr):
    A = self.getAttributes(attr, _spanAttrMap)
    if 'style' in A:
        style = self.findSpanStyle(A.pop('style'))
        D = {}
        for k in 'fontName fontSize textColor backColor'.split():
            v = getattr(style, k, self)
            if v is self:
                continue
            D[k] = v
        D.update(A)
        A = D
    if 'fontName' in A:
        A['fontName'], A['bold'], A['italic'] = ps2tt(A['fontName'])
    self._push('span', **A)