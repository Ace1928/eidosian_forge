from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.BoundFunction)
def handle_bound_function(dmm, ty):
    return dmm[ty.this]