from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def operator_enter(self, new_prec):
    old_prec = self.precedence[-1]
    if old_prec > new_prec:
        self.put(u'(')
    self.precedence.append(new_prec)