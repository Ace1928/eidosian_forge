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
def check_text(text, p=_parser):
    print('##########')
    text = cleanBlockQuotedText(text)
    l, rv, bv = p.parse(text, style)
    if rv is None:
        for l in _parser.errors:
            print(l)
    else:
        print('ParaStyle', l.fontName, l.fontSize, l.textColor)
        for l in rv:
            sys.stdout.write(l.fontName, l.fontSize, l.textColor, l.bold, l.rise, '|%s|' % l.text[:25])
            if hasattr(l, 'cbDefn'):
                print('cbDefn', getattr(l.cbDefn, 'name', ''), getattr(l.cbDefn, 'label', ''), l.cbDefn.kind)
            else:
                print()