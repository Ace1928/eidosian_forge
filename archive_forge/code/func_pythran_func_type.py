from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def pythran_func_type(func, args):
    args = ','.join(('std::declval<%s>()' % pythran_type(a.type) for a in args))
    return 'decltype(%s{}(%s))' % (pythran_functor(func), args)