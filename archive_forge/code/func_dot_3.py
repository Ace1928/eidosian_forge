import contextlib
import warnings
from llvmlite import ir
import numpy as np
import operator
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError, \
from .arrayobj import make_array, _empty_nd_impl, array_copy
from numba.np import numpy_support as np_support
@overload(np.dot)
def dot_3(left, right, out):
    """
    np.dot(a, b, out)
    """
    if isinstance(left, types.Array) and isinstance(right, types.Array) and isinstance(out, types.Array):

        @intrinsic
        def _impl(typingcontext, left, right, out):

            def codegen(context, builder, sig, args):
                ensure_blas()
                with make_contiguous(context, builder, sig, args) as (sig, args):
                    ndims = set((x.ndim for x in sig.args[:2]))
                    if ndims == {2}:
                        return dot_3_mm(context, builder, sig, args)
                    elif ndims == {1, 2}:
                        return dot_3_vm(context, builder, sig, args)
                    else:
                        raise AssertionError('unreachable')
            if left.dtype != right.dtype or left.dtype != out.dtype:
                raise TypingError('np.dot() arguments must all have the same dtype')
            return (signature(out, left, right, out), codegen)
        if left.layout not in 'CF' or right.layout not in 'CF' or out.layout not in 'CF':
            warnings.warn('np.vdot() is faster on contiguous arrays, called on %s' % ((left, right),), NumbaPerformanceWarning)
        return lambda left, right, out: _impl(left, right, out)