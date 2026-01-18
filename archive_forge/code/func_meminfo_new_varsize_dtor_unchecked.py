import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def meminfo_new_varsize_dtor_unchecked(self, builder, size, dtor):
    """
        Like meminfo_new_varsize() but also set the destructor for
        cleaning up references to objects inside the allocation.

        A pointer to the MemInfo is returned.

        Returns NULL to indicate error/failure to allocate.
        """
    self._require_nrt()
    mod = builder.module
    fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t, cgutils.voidptr_t])
    fn = cgutils.get_or_insert_function(mod, fnty, 'NRT_MemInfo_new_varsize_dtor')
    return builder.call(fn, [size, dtor])