from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def operator_exit(self):
    old_prec, new_prec = self.precedence[-2:]
    if old_prec > new_prec:
        self.put(u')')
    self.precedence.pop()