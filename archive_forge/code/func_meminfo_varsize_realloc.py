import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
@_check_null_result
def meminfo_varsize_realloc(self, builder, meminfo, size):
    """
        Reallocate a data area allocated by meminfo_new_varsize().
        The new data pointer is returned, for convenience.

        The result of the call is checked and if it is NULL, i.e. allocation
        failed, then a MemoryError is raised.
        """
    return self.meminfo_varsize_realloc_unchecked(builder, meminfo, size)