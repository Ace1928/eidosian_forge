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
def getAttributes(self, attr, attrMap):
    A = {}
    for k, v in attr.items():
        if not self.caseSensitive:
            k = k.lower()
        if k in attrMap:
            j = attrMap[k]
            func = j[1]
            if func is not None:
                v = func(self, v) if isinstance(func, _ExValidate) else func(v)
            A[j[0]] = v
        else:
            self._syntax_error('invalid attribute name %s attrMap=%r' % (k, list(sorted(attrMap.keys()))))
    return A