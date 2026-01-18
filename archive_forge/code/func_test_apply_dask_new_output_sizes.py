from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
@requires_dask
def test_apply_dask_new_output_sizes() -> None:
    ds = xr.Dataset({'foo': (['lon', 'lat'], np.arange(10 * 10).reshape((10, 10)))})
    ds['bar'] = ds['foo']
    newdims = {'lon_new': 3, 'lat_new': 6}

    def extract(obj):

        def func(da):
            return da[1:4, 1:7]
        return apply_ufunc(func, obj, dask='parallelized', input_core_dims=[['lon', 'lat']], output_core_dims=[['lon_new', 'lat_new']], dask_gufunc_kwargs=dict(output_sizes=newdims))
    expected = extract(ds)
    actual = extract(ds.chunk())
    assert actual.sizes == {'lon_new': 3, 'lat_new': 6}
    assert_identical(expected.chunk(), actual)