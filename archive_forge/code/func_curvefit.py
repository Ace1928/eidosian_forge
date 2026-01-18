from __future__ import annotations
import copy
import datetime
import inspect
import itertools
import math
import sys
import warnings
from collections import defaultdict
from collections.abc import (
from html import escape
from numbers import Number
from operator import methodcaller
from os import PathLike
from typing import IO, TYPE_CHECKING, Any, Callable, Generic, Literal, cast, overload
import numpy as np
import pandas as pd
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftimeindex import CFTimeIndex, _parse_array_of_cftime_strings
from xarray.core import (
from xarray.core import dtypes as xrdtypes
from xarray.core._aggregations import DatasetAggregations
from xarray.core.alignment import (
from xarray.core.arithmetic import DatasetArithmetic
from xarray.core.common import (
from xarray.core.computation import unify_chunks
from xarray.core.coordinates import (
from xarray.core.duck_array_ops import datetime_to_numeric
from xarray.core.indexes import (
from xarray.core.indexing import is_fancy_indexer, map_index_queries
from xarray.core.merge import (
from xarray.core.missing import get_clean_interp_index
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import (
from xarray.core.utils import (
from xarray.core.variable import (
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import array_type, is_chunked_array
from xarray.plot.accessor import DatasetPlotAccessor
from xarray.util.deprecation_helpers import _deprecate_positional_args
def curvefit(self, coords: str | DataArray | Iterable[str | DataArray], func: Callable[..., Any], reduce_dims: Dims=None, skipna: bool=True, p0: Mapping[str, float | DataArray] | None=None, bounds: Mapping[str, tuple[float | DataArray, float | DataArray]] | None=None, param_names: Sequence[str] | None=None, errors: ErrorOptions='raise', kwargs: dict[str, Any] | None=None) -> Self:
    """
        Curve fitting optimization for arbitrary functions.

        Wraps `scipy.optimize.curve_fit` with `apply_ufunc`.

        Parameters
        ----------
        coords : hashable, DataArray, or sequence of hashable or DataArray
            Independent coordinate(s) over which to perform the curve fitting. Must share
            at least one dimension with the calling object. When fitting multi-dimensional
            functions, supply `coords` as a sequence in the same order as arguments in
            `func`. To fit along existing dimensions of the calling object, `coords` can
            also be specified as a str or sequence of strs.
        func : callable
            User specified function in the form `f(x, *params)` which returns a numpy
            array of length `len(x)`. `params` are the fittable parameters which are optimized
            by scipy curve_fit. `x` can also be specified as a sequence containing multiple
            coordinates, e.g. `f((x0, x1), *params)`.
        reduce_dims : str, Iterable of Hashable or None, optional
            Additional dimension(s) over which to aggregate while fitting. For example,
            calling `ds.curvefit(coords='time', reduce_dims=['lat', 'lon'], ...)` will
            aggregate all lat and lon points and fit the specified function along the
            time dimension.
        skipna : bool, default: True
            Whether to skip missing values when fitting. Default is True.
        p0 : dict-like, optional
            Optional dictionary of parameter names to initial guesses passed to the
            `curve_fit` `p0` arg. If the values are DataArrays, they will be appropriately
            broadcast to the coordinates of the array. If none or only some parameters are
            passed, the rest will be assigned initial values following the default scipy
            behavior.
        bounds : dict-like, optional
            Optional dictionary of parameter names to tuples of bounding values passed to the
            `curve_fit` `bounds` arg. If any of the bounds are DataArrays, they will be
            appropriately broadcast to the coordinates of the array. If none or only some
            parameters are passed, the rest will be unbounded following the default scipy
            behavior.
        param_names : sequence of hashable, optional
            Sequence of names for the fittable parameters of `func`. If not supplied,
            this will be automatically determined by arguments of `func`. `param_names`
            should be manually supplied when fitting a function that takes a variable
            number of parameters.
        errors : {"raise", "ignore"}, default: "raise"
            If 'raise', any errors from the `scipy.optimize_curve_fit` optimization will
            raise an exception. If 'ignore', the coefficients and covariances for the
            coordinates where the fitting failed will be NaN.
        **kwargs : optional
            Additional keyword arguments to passed to scipy curve_fit.

        Returns
        -------
        curvefit_results : Dataset
            A single dataset which contains:

            [var]_curvefit_coefficients
                The coefficients of the best fit.
            [var]_curvefit_covariance
                The covariance matrix of the coefficient estimates.

        See Also
        --------
        Dataset.polyfit
        scipy.optimize.curve_fit
        """
    from scipy.optimize import curve_fit
    from xarray.core.alignment import broadcast
    from xarray.core.computation import apply_ufunc
    from xarray.core.dataarray import _THIS_ARRAY, DataArray
    if p0 is None:
        p0 = {}
    if bounds is None:
        bounds = {}
    if kwargs is None:
        kwargs = {}
    reduce_dims_: list[Hashable]
    if not reduce_dims:
        reduce_dims_ = []
    elif isinstance(reduce_dims, str) or not isinstance(reduce_dims, Iterable):
        reduce_dims_ = [reduce_dims]
    else:
        reduce_dims_ = list(reduce_dims)
    if isinstance(coords, str) or isinstance(coords, DataArray) or (not isinstance(coords, Iterable)):
        coords = [coords]
    coords_: Sequence[DataArray] = [self[coord] if isinstance(coord, str) else coord for coord in coords]
    for coord in coords_:
        reduce_dims_ += [c for c in self.dims if coord.equals(self[c])]
    reduce_dims_ = list(set(reduce_dims_))
    preserved_dims = list(set(self.dims) - set(reduce_dims_))
    if not reduce_dims_:
        raise ValueError('No arguments to `coords` were identified as a dimension on the calling object, and no dims were supplied to `reduce_dims`. This would result in fitting on scalar data.')
    for param, guess in p0.items():
        if isinstance(guess, DataArray):
            unexpected = set(guess.dims) - set(preserved_dims)
            if unexpected:
                raise ValueError(f"Initial guess for '{param}' has unexpected dimensions {tuple(unexpected)}. It should only have dimensions that are in data dimensions {preserved_dims}.")
    for param, (lb, ub) in bounds.items():
        for label, bound in zip(('Lower', 'Upper'), (lb, ub)):
            if isinstance(bound, DataArray):
                unexpected = set(bound.dims) - set(preserved_dims)
                if unexpected:
                    raise ValueError(f"{label} bound for '{param}' has unexpected dimensions {tuple(unexpected)}. It should only have dimensions that are in data dimensions {preserved_dims}.")
    if errors not in ['raise', 'ignore']:
        raise ValueError('errors must be either "raise" or "ignore"')
    coords_ = broadcast(*coords_)
    coords_ = [coord.broadcast_like(self, exclude=preserved_dims) for coord in coords_]
    n_coords = len(coords_)
    params, func_args = _get_func_args(func, param_names)
    param_defaults, bounds_defaults = _initialize_curvefit_params(params, p0, bounds, func_args)
    n_params = len(params)

    def _wrapper(Y, *args, **kwargs):
        coords__ = args[:n_coords]
        p0_ = args[n_coords + 0 * n_params:n_coords + 1 * n_params]
        lb = args[n_coords + 1 * n_params:n_coords + 2 * n_params]
        ub = args[n_coords + 2 * n_params:]
        x = np.vstack([c.ravel() for c in coords__])
        y = Y.ravel()
        if skipna:
            mask = np.all([np.any(~np.isnan(x), axis=0), ~np.isnan(y)], axis=0)
            x = x[:, mask]
            y = y[mask]
            if not len(y):
                popt = np.full([n_params], np.nan)
                pcov = np.full([n_params, n_params], np.nan)
                return (popt, pcov)
        x = np.squeeze(x)
        try:
            popt, pcov = curve_fit(func, x, y, p0=p0_, bounds=(lb, ub), **kwargs)
        except RuntimeError:
            if errors == 'raise':
                raise
            popt = np.full([n_params], np.nan)
            pcov = np.full([n_params, n_params], np.nan)
        return (popt, pcov)
    result = type(self)()
    for name, da in self.data_vars.items():
        if name is _THIS_ARRAY:
            name = ''
        else:
            name = f'{str(name)}_'
        input_core_dims = [reduce_dims_ for _ in range(n_coords + 1)]
        input_core_dims.extend([[] for _ in range(3 * n_params)])
        popt, pcov = apply_ufunc(_wrapper, da, *coords_, *param_defaults.values(), *[b[0] for b in bounds_defaults.values()], *[b[1] for b in bounds_defaults.values()], vectorize=True, dask='parallelized', input_core_dims=input_core_dims, output_core_dims=[['param'], ['cov_i', 'cov_j']], dask_gufunc_kwargs={'output_sizes': {'param': n_params, 'cov_i': n_params, 'cov_j': n_params}}, output_dtypes=(np.float64, np.float64), exclude_dims=set(reduce_dims_), kwargs=kwargs)
        result[name + 'curvefit_coefficients'] = popt
        result[name + 'curvefit_covariance'] = pcov
    result = result.assign_coords({'param': params, 'cov_i': params, 'cov_j': params})
    result.attrs = self.attrs.copy()
    return result