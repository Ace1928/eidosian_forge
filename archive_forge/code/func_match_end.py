import codecs
import re
from mako import exceptions
from mako import parsetree
from mako.pygen import adjust_whitespace
def match_end(self):
    match = self.match('\\Z', re.S)
    if not match:
        return False
    string = match.group()
    if string:
        return string
    else:
        return True