import codecs
import re
from mako import exceptions
from mako import parsetree
from mako.pygen import adjust_whitespace
def match_expression(self):
    match = self.match('\\${')
    if not match:
        return False
    line, pos = (self.matched_lineno, self.matched_charpos)
    text, end = self.parse_until_text(True, '\\|', '}')
    if end == '|':
        escapes, end = self.parse_until_text(True, '}')
    else:
        escapes = ''
    text = text.replace('\r\n', '\n')
    self.append_node(parsetree.Expression, text, escapes.strip(), lineno=line, pos=pos)
    return True