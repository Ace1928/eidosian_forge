from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type
def generate_assignment_code(self, rhs, code, overloaded_assignment=False):
    if self.type.is_pyobject:
        rhs.make_owned_reference(code)
        if not self.lhs_of_first_assignment:
            code.put_decref(self.result(), self.ctype())
    code.putln('%s = %s;' % (self.result(), rhs.result() if overloaded_assignment else rhs.result_as(self.ctype())))
    rhs.generate_post_assignment_code(code)
    rhs.free_temps(code)