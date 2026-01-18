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
def start_bullet(self, attr):
    if hasattr(self, 'bFragList'):
        self._syntax_error('only one <bullet> tag allowed')
    self.bFragList = []
    frag = self._initial_frag(attr, _bulletAttrMap, 1)
    frag.isBullet = 1
    frag.__tag__ = 'bullet'
    self._stack.append(frag)