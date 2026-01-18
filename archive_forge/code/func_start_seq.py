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
def start_seq(self, attr):
    if 'template' in attr:
        templ = attr['template']
        self.handle_data(templ % self._seq)
        return
    elif 'id' in attr:
        id = attr['id']
    else:
        id = None
    increment = attr.get('inc', None)
    if not increment:
        output = self._seq.nextf(id)
    elif increment.lower() == 'no':
        output = self._seq.thisf(id)
    else:
        incr = int(increment)
        output = self._seq.thisf(id)
        self._seq.reset(id, self._seq._this() + incr)
    self.handle_data(output)