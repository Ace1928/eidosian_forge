import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import (
from numba.core.analysis import (
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
def check_reduce_func(func_ir, func_var):
    """Checks the function at func_var in func_ir to make sure it's amenable
    for inlining. Returns the function itself"""
    reduce_func = guard(get_definition, func_ir, func_var)
    if reduce_func is None:
        raise ValueError('Reduce function cannot be found for njit                             analysis')
    if isinstance(reduce_func, (ir.FreeVar, ir.Global)):
        if not isinstance(reduce_func.value, numba.core.registry.CPUDispatcher):
            raise ValueError('Invalid reduction function')
        reduce_func = reduce_func.value.py_func
    elif not (hasattr(reduce_func, 'code') or hasattr(reduce_func, '__code__')):
        raise ValueError('Invalid reduction function')
    f_code = reduce_func.code if hasattr(reduce_func, 'code') else reduce_func.__code__
    if not f_code.co_argcount == 2:
        raise TypeError('Reduction function should take 2 arguments')
    return reduce_func