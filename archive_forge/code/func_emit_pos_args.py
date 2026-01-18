from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def emit_pos_args(self, node):
    if node is None:
        return
    if isinstance(node, AddNode):
        self.emit_pos_args(node.operand1)
        self.emit_pos_args(node.operand2)
    elif isinstance(node, TupleNode):
        for expr in node.subexpr_nodes():
            self.visit(expr)
            self.put(u', ')
    elif isinstance(node, AsTupleNode):
        self.put('*')
        self.visit(node.arg)
        self.put(u', ')
    else:
        self.visit(node)
        self.put(u', ')