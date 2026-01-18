import builtins
import unittest
from numbers import Number
from functools import wraps
import numpy as np
from llvmlite import ir
import numba
from numba import njit, typeof, objmode
from numba.core import cgutils, types, typing
from numba.core.pythonapi import box
from numba.core.errors import TypingError
from numba.core.registry import cpu_target
from numba.extending import (intrinsic, lower_builtin, overload_classmethod,
from numba.np import numpy_support
from numba.tests.support import TestCase, MemoryLeakMixin
@lower_builtin(MyArray, types.UniTuple, types.DType, types.Array)
def impl_myarray(context, builder, sig, args):
    from numba.np.arrayobj import make_array, populate_array
    srcaryty = sig.args[-1]
    shape, dtype, buf = args
    srcary = make_array(srcaryty)(context, builder, value=buf)
    retary = make_array(sig.return_type)(context, builder)
    populate_array(retary, data=srcary.data, shape=srcary.shape, strides=srcary.strides, itemsize=srcary.itemsize, meminfo=srcary.meminfo)
    ret = retary._getvalue()
    context.nrt.incref(builder, sig.return_type, ret)
    return ret