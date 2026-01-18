from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
@cython.ccall
def is_pythran_expr(type_):
    return type_.is_pythran_expr