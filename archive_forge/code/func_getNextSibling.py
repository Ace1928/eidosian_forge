from __future__ import absolute_import, division, unicode_literals
from six import text_type
from collections import OrderedDict
from lxml import etree
from ..treebuilders.etree import tag_regexp
from . import base
from .. import _ihatexml
def getNextSibling(self, node):
    if isinstance(node, tuple):
        node, key = node
        assert key in ('text', 'tail'), 'Text nodes are text or tail, found %s' % key
        if key == 'text':
            if len(node):
                return node[0]
            else:
                return None
        else:
            return node.getnext()
    return (node, 'tail') if node.tail else node.getnext()