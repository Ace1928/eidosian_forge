from __future__ import annotations
from itertools import combinations, permutations
from typing import cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import _parse_array_of_cftime_strings
from xarray.core.types import InterpOptions
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
@requires_scipy
@requires_dask
@pytest.mark.parametrize('method', ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'])
@pytest.mark.parametrize('chunked', [True, False])
@pytest.mark.parametrize('data_ndim,interp_ndim,nscalar', [(data_ndim, interp_ndim, nscalar) for data_ndim in range(1, 4) for interp_ndim in range(1, data_ndim + 1) for nscalar in range(0, interp_ndim + 1)])
def test_interpolate_chunk_1d(method: InterpOptions, data_ndim, interp_ndim, nscalar, chunked: bool) -> None:
    """Interpolate nd array with multiple independent indexers

    It should do a series of 1d interpolation
    """
    x = np.linspace(0, 1, 5)
    y = np.linspace(2, 4, 7)
    z = np.linspace(-0.5, 0.5, 11)
    da = xr.DataArray(data=np.sin(x[:, np.newaxis, np.newaxis]) * np.cos(y[:, np.newaxis]) * np.exp(z), coords=[('x', x), ('y', y), ('z', z)])
    kwargs = {'fill_value': 'extrapolate'}
    for data_dims in permutations(da.dims, data_ndim):
        da = da.isel({dim: len(da.coords[dim]) // 2 for dim in da.dims if dim not in data_dims})
        da = da.chunk(chunks={dim: i + 1 for i, dim in enumerate(da.dims)})
        for interp_dims in permutations(da.dims, interp_ndim):
            for scalar_dims in combinations(interp_dims, nscalar):
                dest = {}
                for dim in interp_dims:
                    if dim in scalar_dims:
                        dest[dim] = 0.5 * (da.coords[dim][0] + da.coords[dim][-1])
                    else:
                        before = 2 * da.coords[dim][0] - da.coords[dim][1]
                        after = 2 * da.coords[dim][-1] - da.coords[dim][-2]
                        dest[dim] = cast(xr.DataArray, np.linspace(before.item(), after.item(), len(da.coords[dim]) * 13))
                        if chunked:
                            dest[dim] = xr.DataArray(data=dest[dim], dims=[dim])
                            dest[dim] = dest[dim].chunk(2)
                actual = da.interp(method=method, **dest, kwargs=kwargs)
                expected = da.compute().interp(method=method, **dest, kwargs=kwargs)
                assert_identical(actual, expected)
                break
            break
        break