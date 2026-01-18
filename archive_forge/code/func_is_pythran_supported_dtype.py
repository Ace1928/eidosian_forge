from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
@cython.ccall
def is_pythran_supported_dtype(type_):
    if isinstance(type_, CTypedefType):
        return is_pythran_supported_type(type_.typedef_base_type)
    return type_.is_numeric