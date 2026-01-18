import collections
import contextlib
import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.misc import quicksort
from numba.cpython import slicing
from numba.core.errors import NumbaValueError, TypingError
from numba.core.extending import overload, overload_method, intrinsic
def gen_operator_impl(op, impl):

    @intrinsic
    def _set_operator_intr(typingctx, a, b):
        sig = a(a, b)

        def codegen(context, builder, sig, args):
            assert sig.return_type == sig.args[0]
            impl(context, builder, sig, args)
            return impl_ret_borrowed(context, builder, sig.args[0], args[0])
        return (sig, codegen)

    @overload(op)
    def _ol_set_operator(a, b):
        check_all_set(a, b)
        return lambda a, b: _set_operator_intr(a, b)