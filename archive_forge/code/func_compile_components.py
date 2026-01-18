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
@memoize
def compile_components(agg, schema, glyph, *, antialias=False, cuda=False, partitioned=False):
    """Given an ``Aggregation`` object and a schema, return 5 sub-functions
    and information on how to perform the second stage aggregation if
    antialiasing is requested,

    Parameters
    ----------
    agg : Aggregation
        The expression describing the aggregation(s) to be computed.

    schema : DataShape
        Columns and dtypes in the source dataset.

    glyph : Glyph
        The glyph to render.

    antialias : bool
        Whether to render using antialiasing.

    cuda : bool
        Whether to render using CUDA (on the GPU) or CPU.

    partitioned : bool
        Whether the source dataset is partitioned using dask.

    Returns
    -------
    A tuple of the following:

    ``create(shape)``
        Function that takes the aggregate shape, and returns a tuple of
        initialized numpy arrays.

    ``info(df, canvas_shape)``
        Function that takes a dataframe, and returns preprocessed 1D numpy
        arrays of the needed columns.

    ``append(i, x, y, *aggs_and_cols)``
        Function that appends the ``i``th row of the table to the ``(x, y)``
        bin, given the base arrays and columns in ``aggs_and_cols``. This does
        the bulk of the work.

    ``combine(base_tuples)``
        Function that combines a list of base tuples into a single base tuple.
        This forms the reducing step in a reduction tree.

    ``finalize(aggs, cuda)``
        Function that is given a tuple of base numpy arrays and returns the
        finalized ``DataArray`` or ``Dataset``.

    ``antialias_stage_2``
        If using antialiased lines this is a tuple of the ``AntialiasCombination``
        values corresponding to the aggs. If not using antialiased lines then
        this is ``False``.

    ``antialias_stage_2_funcs``
        If using antialiased lines which require a second stage combine, this
        is a tuple of the three combine functions which are the accumulate,
        clear and copy_back functions. If not using antialiased lines then this
        is ``None``.

    ``column_names``
        Names of DataFrame columns or DataArray variables that are used by the
        agg.
    """
    reds = list(traverse_aggregation(agg))
    bases = list(unique(concat((r._build_bases(cuda, partitioned) for r in reds))))
    dshapes = [b.out_dshape(schema, antialias, cuda, partitioned) for b in bases]
    if antialias:
        self_intersect, antialias_stage_2 = make_antialias_stage_2(reds, bases)
        if cuda:
            import cupy
            array_module = cupy
        else:
            array_module = np
        antialias_stage_2 = antialias_stage_2(array_module)
        antialias_stage_2_funcs = make_antialias_stage_2_functions(antialias_stage_2, bases, cuda, partitioned)
    else:
        self_intersect = False
        antialias_stage_2 = False
        antialias_stage_2_funcs = None
    calls = [_get_call_tuples(b, d, schema, cuda, antialias, self_intersect, partitioned) for b, d in zip(bases, dshapes)]
    cols = list(concat(pluck(2, calls)))
    nan_check_cols = list((c[3] for c in calls if c[3] is not None))
    cols = list(unique(cols + nan_check_cols))
    temps = list(pluck(4, calls))
    combine_temps = list(pluck(5, calls))
    create = make_create(bases, dshapes, cuda)
    append, any_uses_cuda_mutex = make_append(bases, cols, calls, glyph, antialias)
    info = make_info(cols, cuda, any_uses_cuda_mutex)
    combine = make_combine(bases, dshapes, temps, combine_temps, antialias, cuda, partitioned)
    finalize = make_finalize(bases, agg, schema, cuda, partitioned)
    column_names = [c.column for c in cols if c.column != SpecialColumn.RowIndex]
    return (create, info, append, combine, finalize, antialias_stage_2, antialias_stage_2_funcs, column_names)