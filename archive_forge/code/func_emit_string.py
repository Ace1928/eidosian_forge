from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def emit_string(self, node, prefix=u''):
    repr_val = repr(node.value)
    if repr_val[0] in 'ub':
        repr_val = repr_val[1:]
    self.put(u'%s%s' % (prefix, repr_val))