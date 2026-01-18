from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type
def generate_execution_code(self, code):
    self.setup_temp_expr(code)
    self.body.generate_execution_code(code)
    self.teardown_temp_expr(code)