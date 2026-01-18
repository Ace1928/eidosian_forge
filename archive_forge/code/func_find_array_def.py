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
def find_array_def(arr):
    """Find numpy array definition such as
            arr = numba.unsafe.ndarray.empty_inferred(...).
        If it is arr = b[...], find array definition of b recursively.
        """
    arr_def = get_definition(func_ir, arr)
    _make_debug_print('find_array_def')(arr, arr_def)
    if isinstance(arr_def, ir.Expr):
        if guard(_find_unsafe_empty_inferred, func_ir, arr_def):
            return arr_def
        elif arr_def.op == 'getitem':
            return find_array_def(arr_def.value)
    raise GuardException