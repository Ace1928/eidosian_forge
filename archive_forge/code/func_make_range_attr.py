import operator
from numba import prange
from numba.core import types, cgutils, errors
from numba.cpython.listobj import ListIterInstance
from numba.np.arrayobj import make_array
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, overload_attribute, register_jitable
from numba.parfors.parfor import internal_prange
def make_range_attr(index, attribute):

    @intrinsic
    def rangetype_attr_getter(typingctx, a):
        if isinstance(a, types.RangeType):

            def codegen(context, builder, sig, args):
                val, = args
                items = cgutils.unpack_tuple(builder, val, 3)
                return impl_ret_untracked(context, builder, sig.return_type, items[index])
            return (signature(a.dtype, a), codegen)

    @overload_attribute(types.RangeType, attribute)
    def range_attr(rnge):

        def get(rnge):
            return rangetype_attr_getter(rnge)
        return get