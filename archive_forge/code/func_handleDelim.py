from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def handleDelim(self, cc, block):
    """Handle a delimiter marker for emphasis or a quote."""
    res = self.scanDelims(cc)
    if not res:
        return False
    numdelims = res.get('numdelims')
    startpos = self.pos
    self.pos += numdelims
    if cc == "'":
        contents = '’'
    elif cc == '"':
        contents = '“'
    else:
        contents = self.subject[startpos:self.pos]
    node = text(contents)
    block.append_child(node)
    self.delimiters = {'cc': cc, 'numdelims': numdelims, 'origdelims': numdelims, 'node': node, 'previous': self.delimiters, 'next': None, 'can_open': res.get('can_open'), 'can_close': res.get('can_close')}
    if self.delimiters['previous'] is not None:
        self.delimiters['previous']['next'] = self.delimiters
    return True