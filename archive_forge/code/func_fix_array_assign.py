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
def fix_array_assign(stmt):
    """For assignment like lhs[idx] = rhs, where both lhs and rhs are
        arrays, do the following:
        1. find the definition of rhs, which has to be a call to
           numba.unsafe.ndarray.empty_inferred
        2. find the source array creation for lhs, insert an extra dimension of
           size of b.
        3. replace the definition of
           rhs = numba.unsafe.ndarray.empty_inferred(...) with rhs = lhs[idx]
        """
    require(isinstance(stmt, ir.SetItem))
    require(isinstance(stmt.value, ir.Var))
    debug_print = _make_debug_print('fix_array_assign')
    debug_print('found SetItem: ', stmt)
    lhs = stmt.target
    lhs_def = find_array_def(lhs)
    debug_print('found lhs_def: ', lhs_def)
    rhs_def = get_definition(func_ir, stmt.value)
    debug_print('found rhs_def: ', rhs_def)
    require(isinstance(rhs_def, ir.Expr))
    if rhs_def.op == 'cast':
        rhs_def = get_definition(func_ir, rhs_def.value)
        require(isinstance(rhs_def, ir.Expr))
    require(_find_unsafe_empty_inferred(func_ir, rhs_def))
    dim_def = get_definition(func_ir, rhs_def.args[0])
    require(isinstance(dim_def, ir.Expr) and dim_def.op == 'build_tuple')
    debug_print('dim_def = ', dim_def)
    extra_dims = [get_definition(func_ir, x, lhs_only=True) for x in dim_def.items]
    debug_print('extra_dims = ', extra_dims)
    size_tuple_def = get_definition(func_ir, lhs_def.args[0])
    require(isinstance(size_tuple_def, ir.Expr) and size_tuple_def.op == 'build_tuple')
    debug_print('size_tuple_def = ', size_tuple_def)
    extra_dims = fix_dependencies(size_tuple_def, extra_dims)
    size_tuple_def.items += extra_dims
    rhs_def.op = 'getitem'
    rhs_def.fn = operator.getitem
    rhs_def.value = get_definition(func_ir, lhs, lhs_only=True)
    rhs_def.index = stmt.index
    del rhs_def._kws['func']
    del rhs_def._kws['args']
    del rhs_def._kws['vararg']
    del rhs_def._kws['kws']
    return True