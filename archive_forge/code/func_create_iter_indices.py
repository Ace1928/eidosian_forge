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
def create_iter_indices(self):
    intpty = self.context.get_value_type(types.intp)
    ZERO = ir.Constant(ir.IntType(intpty.width), 0)
    indices = []
    for i in range(self.ndim):
        x = cgutils.alloca_once(self.builder, ir.IntType(intpty.width))
        self.builder.store(ZERO, x)
        indices.append(x)
    return _ArrayIndexingHelper(self, indices)