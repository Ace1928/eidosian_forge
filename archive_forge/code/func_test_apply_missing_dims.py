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
def test_apply_missing_dims() -> None:

    def add_one(a, core_dims, on_missing_core_dim):
        return apply_ufunc(lambda x: x + 1, a, input_core_dims=core_dims, output_core_dims=core_dims, on_missing_core_dim=on_missing_core_dim)
    array = np.arange(6).reshape(2, 3)
    variable = xr.Variable(['x', 'y'], array)
    variable_no_y = xr.Variable(['x', 'z'], array)
    ds = xr.Dataset({'x_y': variable, 'x_z': variable_no_y})
    assert_identical(add_one(ds[['x_y']], core_dims=[['y']], on_missing_core_dim='raise'), ds[['x_y']] + 1)
    with pytest.raises(ValueError):
        add_one(ds, core_dims=[['y']], on_missing_core_dim='raise')
    assert_identical(add_one(ds, core_dims=[['y']], on_missing_core_dim='drop'), (ds + 1).drop_vars('x_z'))
    copy_result = add_one(ds, core_dims=[['y']], on_missing_core_dim='copy')
    assert_identical(copy_result['x_y'], (ds + 1)['x_y'])
    assert_identical(copy_result['x_z'], ds['x_z'])

    def sum_add(a, b, core_dims, on_missing_core_dim):
        return apply_ufunc(lambda a, b, axis=None: a.sum(axis) + b.sum(axis), a, b, input_core_dims=core_dims, on_missing_core_dim=on_missing_core_dim)
    assert_identical(sum_add(ds[['x_y']], ds[['x_y']], core_dims=[['x', 'y'], ['x', 'y']], on_missing_core_dim='raise'), ds[['x_y']].sum() * 2)
    with pytest.raises(ValueError, match=".*Missing core dims \\{'y'\\} from arg number 1 on a variable named `x_z`:\\n.*<xarray.Variable \\(x: 2, z: "):
        sum_add(ds[['x_z']], ds[['x_z']], core_dims=[['x', 'y'], ['x', 'z']], on_missing_core_dim='raise')
    with pytest.raises(ValueError, match=".*Missing core dims \\{'y'\\} from arg number 2 on a variable named `x_z`:\\n.*<xarray.Variable \\(x: 2, z: "):
        sum_add(ds[['x_z']], ds[['x_z']], core_dims=[['x', 'z'], ['x', 'y']], on_missing_core_dim='raise')
    assert_identical(sum_add(ds[['x_z']], ds[['x_z']], core_dims=[['x', 'y'], ['x', 'y']], on_missing_core_dim='drop'), ds[[]])
    assert_identical(sum_add(ds[['x_z']], ds[['x_z']], core_dims=[['x', 'y'], ['x', 'y']], on_missing_core_dim='copy'), ds[['x_z']])
    assert_identical(sum_add(ds[['x_y', 'x_y']], ds[['x_y', 'x_y']], core_dims=[['x', 'y'], ['x', 'y']], on_missing_core_dim='raise'), ds[['x_y', 'x_y']].sum() * 2)
    assert_identical(sum_add(ds, ds, core_dims=[['x', 'y'], ['x', 'y']], on_missing_core_dim='drop'), ds[['x_y']].sum() * 2)
    assert_identical(sum_add(ds.assign(x_t=ds['x_z'])[['x_y', 'x_t']], ds.assign(x_t=ds['x_y'])[['x_y', 'x_t']], core_dims=[['x', 'y'], ['x', 'y']], on_missing_core_dim='drop'), ds[['x_y']].sum() * 2)
    assert_identical(sum_add(ds.assign(x_t=ds['x_y'])[['x_y', 'x_t']], ds.assign(x_t=ds['x_z'])[['x_y', 'x_t']], core_dims=[['x', 'y'], ['x', 'y']], on_missing_core_dim='drop'), ds[['x_y']].sum() * 2)
    assert_identical(sum_add(ds.assign(x_t=ds['x_y'])[['x_y', 'x_t']], ds.assign(x_t=ds['x_z'])[['x_y', 'x_t']], core_dims=[['x', 'y'], ['x', 'y']], on_missing_core_dim='copy'), ds.drop_vars('x_z').assign(x_y=30, x_t=ds['x_y']))