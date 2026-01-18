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
def handle_data(self, data):
    """Creates an intermediate representation of string segments."""
    frag = copy.copy(self._stack[-1])
    if hasattr(frag, 'cbDefn'):
        kind = frag.cbDefn.kind
        if data:
            self._syntax_error('Only empty <%s> tag allowed' % kind)
    elif hasattr(frag, '_selfClosingTag'):
        if data != '':
            self._syntax_error('No content allowed in %s tag' % frag._selfClosingTag)
        return
    elif frag.greek:
        frag.fontName = 'symbol'
        data = _greekConvert(data)
    frag.fontName = tt2ps(frag.fontName, frag.bold, frag.italic)
    frag.text = data
    if hasattr(frag, 'isBullet'):
        delattr(frag, 'isBullet')
        self.bFragList.append(frag)
    else:
        self.fragList.append(frag)