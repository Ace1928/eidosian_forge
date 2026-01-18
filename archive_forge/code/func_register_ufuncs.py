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
def register_ufuncs(ufuncs, lower):
    kernels = {}
    for ufunc in ufuncs:
        db_func = _ufunc_db_function(ufunc)
        kernels[ufunc] = register_ufunc_kernel(ufunc, db_func, lower)
    for _op_map in (npydecl.NumpyRulesUnaryArrayOperator._op_map, npydecl.NumpyRulesArrayOperator._op_map):
        for operator, ufunc_name in _op_map.items():
            ufunc = getattr(np, ufunc_name)
            kernel = kernels[ufunc]
            if ufunc.nin == 1:
                register_unary_operator_kernel(operator, ufunc, kernel, lower)
            elif ufunc.nin == 2:
                register_binary_operator_kernel(operator, ufunc, kernel, lower)
            else:
                raise RuntimeError("There shouldn't be any non-unary or binary operators")
    for _op_map in (npydecl.NumpyRulesInplaceArrayOperator._op_map,):
        for operator, ufunc_name in _op_map.items():
            ufunc = getattr(np, ufunc_name)
            kernel = kernels[ufunc]
            if ufunc.nin == 1:
                register_unary_operator_kernel(operator, ufunc, kernel, lower, inplace=True)
            elif ufunc.nin == 2:
                register_binary_operator_kernel(operator, ufunc, kernel, lower, inplace=True)
            else:
                raise RuntimeError("There shouldn't be any non-unary or binary operators")