from __future__ import absolute_import
from .Errors import CompileError, error
from . import ExprNodes
from .ExprNodes import IntNode, NameNode, AttributeNode
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .UtilityCode import CythonUtilityCode
from . import Buffer
from . import PyrexTypes
from . import ModuleNode
def put_assign_to_memviewslice(lhs_cname, rhs, rhs_cname, memviewslicetype, code, have_gil=False, first_assignment=False):
    if lhs_cname == rhs_cname:
        code.putln('/* memoryview self assignment no-op */')
        return
    if not first_assignment:
        code.put_xdecref(lhs_cname, memviewslicetype, have_gil=have_gil)
    if not rhs.result_in_temp():
        rhs.make_owned_memoryviewslice(code)
    code.putln('%s = %s;' % (lhs_cname, rhs_cname))