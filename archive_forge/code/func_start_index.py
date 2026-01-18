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
def start_index(self, attr):
    attr = self.getAttributes(attr, _indexAttrMap)
    defn = ABag()
    if 'item' in attr:
        label = attr['item']
    else:
        self._syntax_error('<index> needs at least an item attribute')
    if 'name' in attr:
        name = attr['name']
    else:
        name = DEFAULT_INDEX_NAME
    format = attr.get('format', None)
    if format is not None and format not in ('123', 'I', 'i', 'ABC', 'abc'):
        raise ValueError('index tag format is %r not valid 123 I i ABC or abc' % offset)
    offset = attr.get('offset', None)
    if offset is not None:
        try:
            offset = int(offset)
        except:
            raise ValueError('index tag offset is %r not an int' % offset)
    defn.label = encode_label((label, format, offset))
    defn.name = name
    defn.kind = 'index'
    self._push('index', cbDefn=defn)
    self.handle_data('')
    self._pop('index')