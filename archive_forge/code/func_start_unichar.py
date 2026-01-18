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
def start_unichar(self, attr):
    if 'name' in attr:
        if 'code' in attr:
            self._syntax_error('<unichar/> invalid with both name and code attributes')
        try:
            v = unicodedata.lookup(attr['name'])
        except KeyError:
            self._syntax_error('<unichar/> invalid name attribute\n"%s"' % ascii(attr['name']))
            v = '\x00'
    elif 'code' in attr:
        try:
            v = attr['code'].lower()
            if v.startswith('0x'):
                v = int(v, 16)
            else:
                v = int(v, 0)
            v = chr(v)
        except:
            self._syntax_error('<unichar/> invalid code attribute %s' % ascii(attr['code']))
            v = '\x00'
    else:
        v = None
        if attr:
            self._syntax_error('<unichar/> invalid attribute %s' % list(attr.keys())[0])
    if v is not None:
        self.handle_data(v)
    self._push('unichar', _selfClosingTag='unichar')