import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def allocate_unchecked(self, builder, size):
    """
        Low-level allocate a new memory area of `size` bytes. Returns NULL to
        indicate error/failure to allocate.
        """
    self._require_nrt()
    mod = builder.module
    fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t])
    fn = cgutils.get_or_insert_function(mod, fnty, 'NRT_Allocate')
    fn.return_value.add_attribute('noalias')
    return builder.call(fn, [size])