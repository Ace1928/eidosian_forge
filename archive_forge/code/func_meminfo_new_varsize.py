import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
@_check_null_result
def meminfo_new_varsize(self, builder, size):
    """
        Allocate a MemInfo pointing to a variable-sized data area.  The area
        is separately allocated (i.e. two allocations are made) so that
        re-allocating it doesn't change the MemInfo's address.

        A pointer to the MemInfo is returned.

        The result of the call is checked and if it is NULL, i.e. allocation
        failed, then a MemoryError is raised.
        """
    return self.meminfo_new_varsize_unchecked(builder, size)