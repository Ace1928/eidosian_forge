from __future__ import absolute_import, division, unicode_literals
from six import text_type
from collections import OrderedDict
from lxml import etree
from ..treebuilders.etree import tag_regexp
from . import base
from .. import _ihatexml
def getFirstChild(self, node):
    assert not isinstance(node, tuple), 'Text nodes have no children'
    assert len(node) or node.text, 'Node has no children'
    if node.text:
        return (node, 'text')
    else:
        return node[0]