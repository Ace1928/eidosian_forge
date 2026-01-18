from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def parseHtmlTag(self, block):
    """Attempt to parse a raw HTML tag."""
    m = self.match(common.reHtmlTag)
    if m is None:
        return False
    else:
        node = Node('html_inline', None)
        node.literal = m
        block.append_child(node)
        return True