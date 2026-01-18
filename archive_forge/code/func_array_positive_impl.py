import math
import sys
import itertools
from collections import namedtuple
import llvmlite.ir as ir
import numpy as np
import operator
from numba.np import arrayobj, ufunc_db, numpy_support
from numba.core.imputils import Registry, impl_ret_new_ref, force_error_model
from numba.core import typing, types, utils, cgutils, callconv
from numba.np.numpy_support import (
from numba.core.typing import npydecl
from numba.core.extending import overload, intrinsic
from numba.core import errors
from numba.cpython import builtins
@registry.lower(operator.pos, types.Array)
def array_positive_impl(context, builder, sig, args):
    """Lowering function for +(array) expressions.  Defined here
    (numba.targets.npyimpl) since the remaining array-operator
    lowering functions are also registered in this module.
    """

    class _UnaryPositiveKernel(_Kernel):

        def generate(self, *args):
            [val] = args
            return val
    return numpy_ufunc_kernel(context, builder, sig, args, np.positive, _UnaryPositiveKernel)