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
def start_ondraw(self, attr):
    defn = ABag()
    if 'name' in attr:
        defn.name = attr['name']
    else:
        self._syntax_error('<onDraw> needs at least a name attribute')
    defn.label = attr.get('label', None)
    defn.kind = 'onDraw'
    self._push('ondraw', cbDefn=defn)
    self.handle_data('')
    self._pop('ondraw')