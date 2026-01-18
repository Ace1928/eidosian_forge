import codecs
import re
from mako import exceptions
from mako import parsetree
from mako.pygen import adjust_whitespace
def match_percent(self):
    match = self.match('(?<=^)(\\s*)%%(%*)', re.M)
    if match:
        self.append_node(parsetree.Text, match.group(1) + '%' + match.group(2))
        return True
    else:
        return False