import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def meminfo_varsize_realloc_unchecked(self, builder, meminfo, size):
    """
        Reallocate a data area allocated by meminfo_new_varsize().
        The new data pointer is returned, for convenience.

        Returns NULL to indicate error/failure to allocate.
        """
    return self._call_varsize_alloc(builder, meminfo, size, 'NRT_MemInfo_varsize_realloc')