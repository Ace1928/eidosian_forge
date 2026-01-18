import operator
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import types
from numba.core.extending import overload_method
@overload_method(types.IntEnumMember, '__hash__')
def intenum_hash(val):

    def hash_impl(val):
        return hash(val.value)
    return hash_impl