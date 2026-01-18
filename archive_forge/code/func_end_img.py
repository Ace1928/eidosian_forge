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
def end_img(self):
    frag = self._stack[-1]
    if not getattr(frag, '_selfClosingTag', ''):
        raise ValueError('Parser failure in <img/>')
    defn = frag.cbDefn = ABag()
    defn.kind = 'img'
    defn.src = getattr(frag, 'src', None)
    defn.image = ImageReader(defn.src)
    size = defn.image.getSize()
    defn.width = getattr(frag, 'width', size[0])
    defn.height = getattr(frag, 'height', size[1])
    defn.valign = getattr(frag, 'valign', 'bottom')
    del frag._selfClosingTag
    self.handle_data('')
    self._pop('img')