from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
def generate_shared_aggregator(func: Callable[..., Scalar], dtype_mapping: dict[np.dtype, np.dtype], is_grouped_kernel: bool, nopython: bool, nogil: bool, parallel: bool):
    """
    Generate a Numba function that loops over the columns 2D object and applies
    a 1D numba kernel over each column.

    Parameters
    ----------
    func : function
        aggregation function to be applied to each column
    dtype_mapping: dict or None
        If not None, maps a dtype to a result dtype.
        Otherwise, will fall back to default mapping.
    is_grouped_kernel: bool, default False
        Whether func operates using the group labels (True)
        or using starts/ends arrays

        If true, you also need to pass the number of groups to this function
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit

    Returns
    -------
    Numba function
    """

    def looper_wrapper(values, start=None, end=None, labels=None, ngroups=None, min_periods: int=0, **kwargs):
        result_dtype = dtype_mapping[values.dtype]
        column_looper = make_looper(func, result_dtype, is_grouped_kernel, nopython, nogil, parallel)
        if is_grouped_kernel:
            result, na_positions = column_looper(values, labels, ngroups, min_periods, *kwargs.values())
        else:
            result, na_positions = column_looper(values, start, end, min_periods, *kwargs.values())
        if result.dtype.kind == 'i':
            for na_pos in na_positions.values():
                if len(na_pos) > 0:
                    result = result.astype('float64')
                    break
        for i, na_pos in na_positions.items():
            if len(na_pos) > 0:
                result[i, na_pos] = np.nan
        return result
    return looper_wrapper