from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def emit_sequence(self, node, parens=(u'', u'')):
    open_paren, close_paren = parens
    items = node.subexpr_nodes()
    self.put(open_paren)
    self.comma_separated_list(items)
    self.put(close_paren)