from __future__ import unicode_literals
import unittest
import commonmark
from commonmark.blocks import Parser
from commonmark.render.html import HtmlRenderer
from commonmark.inlines import InlineParser
from commonmark.node import NodeWalker, Node
def test_dumpAST_orderedlist(self):
    md = '1.'
    ast = Parser().parse(md)
    commonmark.dumpAST(ast)