from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type
def set_temp_expr(self, lazy_temp):
    self.lazy_temp = lazy_temp
    self.temp_expression = lazy_temp.expression