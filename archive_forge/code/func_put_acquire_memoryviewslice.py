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
def put_acquire_memoryviewslice(lhs_cname, lhs_type, lhs_pos, rhs, code, have_gil=False, first_assignment=True):
    """We can avoid decreffing the lhs if we know it is the first assignment"""
    assert rhs.type.is_memoryviewslice
    pretty_rhs = rhs.result_in_temp() or rhs.is_simple()
    if pretty_rhs:
        rhstmp = rhs.result()
    else:
        rhstmp = code.funcstate.allocate_temp(lhs_type, manage_ref=False)
        code.putln('%s = %s;' % (rhstmp, rhs.result_as(lhs_type)))
    put_assign_to_memviewslice(lhs_cname, rhs, rhstmp, lhs_type, code, have_gil=have_gil, first_assignment=first_assignment)
    if not pretty_rhs:
        code.funcstate.release_temp(rhstmp)