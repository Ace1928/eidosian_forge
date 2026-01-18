from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def parseAutolink(self, block):
    """Attempt to parse an autolink (URL or email in pointy brackets)."""
    m = self.match(reEmailAutolink)
    if m:
        dest = m[1:-1]
        node = Node('link', None)
        node.destination = normalize_uri('mailto:' + dest)
        node.title = ''
        node.append_child(text(dest))
        block.append_child(node)
        return True
    else:
        m = self.match(reAutolink)
        if m:
            dest = m[1:-1]
            node = Node('link', None)
            node.destination = normalize_uri(dest)
            node.title = ''
            node.append_child(text(dest))
            block.append_child(node)
            return True
    return False