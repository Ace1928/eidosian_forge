from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def np_func_to_list(func):
    if not func.is_numpy_attribute:
        return []
    return np_func_to_list(func.obj) + [func.attribute]