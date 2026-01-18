from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def parseNewline(self, block):
    """
        Parse a newline.  If it was preceded by two spaces, return a hard
        line break; otherwise a soft line break.
        """
    self.pos += 1
    lastc = block.last_child
    if lastc and lastc.t == 'text' and (lastc.literal[-1] == ' '):
        linebreak = len(lastc.literal) >= 2 and lastc.literal[-2] == ' '
        lastc.literal = re.sub(reFinalSpace, '', lastc.literal)
        if linebreak:
            node = Node('linebreak', None)
        else:
            node = Node('softbreak', None)
        block.append_child(node)
    else:
        block.append_child(Node('softbreak', None))
    self.match(reInitialSpace)
    return True