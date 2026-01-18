from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def include_pythran_generic(env):
    env.add_include_file('pythonic/core.hpp')
    env.add_include_file('pythonic/python/core.hpp')
    env.add_include_file('pythonic/types/bool.hpp')
    env.add_include_file('pythonic/types/ndarray.hpp')
    env.add_include_file('pythonic/numpy/power.hpp')
    env.add_include_file('pythonic/%s/slice.hpp' % pythran_builtins)
    env.add_include_file('<new>')
    for i in (8, 16, 32, 64):
        env.add_include_file('pythonic/types/uint%d.hpp' % i)
        env.add_include_file('pythonic/types/int%d.hpp' % i)
    for t in ('float', 'float32', 'float64', 'set', 'slice', 'tuple', 'int', 'complex', 'complex64', 'complex128'):
        env.add_include_file('pythonic/types/%s.hpp' % t)