import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
def get_unary_impl_for_fn_and_ty(fn, ty):
    tanh_impls = ('tanh', 'tanhf', math.tanh)
    for fname64, fname32, key in unarys + [tanh_impls]:
        if fn == key:
            if ty == float32:
                impl = getattr(libdevice, fname32)
            elif ty == float64:
                impl = getattr(libdevice, fname64)
            return get_lower_unary_impl(key, ty, impl)
    raise RuntimeError(f'Implementation of {fn} for {ty} not found')