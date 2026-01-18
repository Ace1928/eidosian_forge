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
def start_a(self, attributes):
    anchor = 'name' in attributes
    if anchor:
        A = self.getAttributes(attributes, _anchorAttrMap)
        name = A.get('name', None)
        name = name.strip()
        if not name:
            self._syntax_error('<a name="..."/> anchor variant requires non-blank name')
        if len(A) > 1:
            self._syntax_error('<a name="..."/> anchor variant only allows name attribute')
            A = dict(name=A['name'])
        A['_selfClosingTag'] = 'anchor'
        self._push('a', **A)
    else:
        self._handle_link('a', attributes)