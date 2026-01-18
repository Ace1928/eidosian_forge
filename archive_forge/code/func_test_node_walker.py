from __future__ import unicode_literals
import unittest
import commonmark
from commonmark.blocks import Parser
from commonmark.render.html import HtmlRenderer
from commonmark.inlines import InlineParser
from commonmark.node import NodeWalker, Node
def test_node_walker(self):
    node = Node('document', [[1, 1], [0, 0]])
    NodeWalker(node)