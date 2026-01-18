from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
@cython.ccall
def is_pythran_supported_type(type_):
    pythran_supported = ('is_pythran_expr', 'is_int', 'is_numeric', 'is_float', 'is_none', 'is_complex')
    return is_type(type_, pythran_supported) or is_pythran_expr(type_)