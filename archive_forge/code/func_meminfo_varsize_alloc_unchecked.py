import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def meminfo_varsize_alloc_unchecked(self, builder, meminfo, size):
    """
        Allocate a new data area for a MemInfo created by meminfo_new_varsize().
        The new data pointer is returned, for convenience.

        Contrary to realloc(), this always allocates a new area and doesn't
        copy the old data.  This is useful if resizing a container needs
        more than simply copying the data area (e.g. for hash tables).

        The old pointer will have to be freed with meminfo_varsize_free().

        Returns NULL to indicate error/failure to allocate.
        """
    return self._call_varsize_alloc(builder, meminfo, size, 'NRT_MemInfo_varsize_alloc')