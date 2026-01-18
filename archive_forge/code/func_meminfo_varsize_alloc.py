import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
@_check_null_result
def meminfo_varsize_alloc(self, builder, meminfo, size):
    """
        Allocate a new data area for a MemInfo created by meminfo_new_varsize().
        The new data pointer is returned, for convenience.

        Contrary to realloc(), this always allocates a new area and doesn't
        copy the old data.  This is useful if resizing a container needs
        more than simply copying the data area (e.g. for hash tables).

        The old pointer will have to be freed with meminfo_varsize_free().

        The result of the call is checked and if it is NULL, i.e. allocation
        failed, then a MemoryError is raised.
        """
    return self.meminfo_varsize_alloc_unchecked(builder, meminfo, size)