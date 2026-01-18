from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def pythran_get_func_include_file(func):
    func = np_func_to_list(func)
    return 'pythonic/numpy/%s.hpp' % '/'.join(func)