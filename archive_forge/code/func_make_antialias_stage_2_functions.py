from __future__ import annotations
from itertools import count
import logging
from typing import TYPE_CHECKING
from toolz import unique, concat, pluck, get, memoize
from numba import literal_unroll
import numpy as np
import xarray as xr
from .antialias import AntialiasCombination
from .reductions import SpecialColumn, UsesCudaMutex, by, category_codes, summary
from .utils import (isnull, ngjit,
def make_antialias_stage_2_functions(antialias_stage_2, bases, cuda, partitioned):
    aa_combinations, aa_zeroes, aa_n_reductions, aa_categorical = antialias_stage_2
    funcs = [_get_antialias_stage_2_combine_func(comb, zero, n_red, cat) for comb, zero, n_red, cat in zip(aa_combinations, aa_zeroes, aa_n_reductions, aa_categorical)]
    base_is_where = [b.is_where() for b in bases]
    next_base_is_where = base_is_where[1:] + [False]
    namespace = {}
    namespace['literal_unroll'] = literal_unroll
    for func in set(funcs):
        namespace[func.__name__] = func
    names = (f'combine{i}' for i in count())
    lines = ['def aa_stage_2_accumulate(aggs_and_copies, first_pass):', '    if first_pass:', '        for a in literal_unroll(aggs_and_copies):', '            a[1][:] = a[0][:]', '    else:']
    for i, (func, is_where, next_is_where) in enumerate(zip(funcs, base_is_where, next_base_is_where)):
        if is_where:
            where_reduction = bases[i]
            if isinstance(where_reduction, by):
                where_reduction = where_reduction.reduction
            combine = where_reduction._combine_callback(cuda, partitioned, aa_categorical[i])
            name = next(names)
            namespace[name] = combine
            lines.append(f'        {name}(aggs_and_copies[{i}][::-1], aggs_and_copies[{i - 1}][::-1])')
        elif next_is_where:
            pass
        else:
            lines.append(f'        {func.__name__}(aggs_and_copies[{i}][1], aggs_and_copies[{i}][0])')
    code = '\n'.join(lines)
    logger.debug(code)
    exec(code, namespace)
    aa_stage_2_accumulate = ngjit(namespace['aa_stage_2_accumulate'])
    if np.any(np.isnan(aa_zeroes)):
        namespace['nan'] = np.nan
    lines = ['def aa_stage_2_clear(aggs_and_copies):']
    for i, aa_zero in enumerate(aa_zeroes):
        lines.append(f'    aggs_and_copies[{i}][0].fill({aa_zero})')
    code = '\n'.join(lines)
    logger.debug(code)
    exec(code, namespace)
    aa_stage_2_clear = ngjit(namespace['aa_stage_2_clear'])

    @ngjit
    def aa_stage_2_copy_back(aggs_and_copies):
        for agg_and_copy in literal_unroll(aggs_and_copies):
            agg_and_copy[0][:] = agg_and_copy[1][:]
    return (aa_stage_2_accumulate, aa_stage_2_clear, aa_stage_2_copy_back)