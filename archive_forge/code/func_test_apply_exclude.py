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
def test_apply_exclude() -> None:

    def concatenate(objects, dim='x'):

        def func(*x):
            return np.concatenate(x, axis=-1)
        result = apply_ufunc(func, *objects, input_core_dims=[[dim]] * len(objects), output_core_dims=[[dim]], exclude_dims={dim})
        if isinstance(result, (xr.Dataset, xr.DataArray)):
            new_coord = np.concatenate([obj.coords[dim] for obj in objects])
            result.coords[dim] = new_coord
        return result
    arrays = [np.array([1]), np.array([2, 3])]
    variables = [xr.Variable('x', a) for a in arrays]
    data_arrays = [xr.DataArray(v, {'x': c, 'y': ('x', range(len(c)))}) for v, c in zip(variables, [['a'], ['b', 'c']])]
    datasets = [xr.Dataset({'data': data_array}) for data_array in data_arrays]
    expected_array = np.array([1, 2, 3])
    expected_variable = xr.Variable('x', expected_array)
    expected_data_array = xr.DataArray(expected_variable, [('x', list('abc'))])
    expected_dataset = xr.Dataset({'data': expected_data_array})
    assert_identical(expected_array, concatenate(arrays))
    assert_identical(expected_variable, concatenate(variables))
    assert_identical(expected_data_array, concatenate(data_arrays))
    assert_identical(expected_dataset, concatenate(datasets))
    with pytest.raises(ValueError):
        apply_ufunc(identity, variables[0], exclude_dims={'x'})