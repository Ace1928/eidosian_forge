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
def test_apply_input_core_dimension() -> None:

    def first_element(obj, dim):

        def func(x):
            return x[..., 0]
        return apply_ufunc(func, obj, input_core_dims=[[dim]])
    array = np.array([[1, 2], [3, 4]])
    variable = xr.Variable(['x', 'y'], array)
    data_array = xr.DataArray(variable, {'x': ['a', 'b'], 'y': [-1, -2]})
    dataset = xr.Dataset({'data': data_array})
    expected_variable_x = xr.Variable(['y'], [1, 2])
    expected_data_array_x = xr.DataArray(expected_variable_x, {'y': [-1, -2]})
    expected_dataset_x = xr.Dataset({'data': expected_data_array_x})
    expected_variable_y = xr.Variable(['x'], [1, 3])
    expected_data_array_y = xr.DataArray(expected_variable_y, {'x': ['a', 'b']})
    expected_dataset_y = xr.Dataset({'data': expected_data_array_y})
    assert_identical(expected_variable_x, first_element(variable, 'x'))
    assert_identical(expected_variable_y, first_element(variable, 'y'))
    assert_identical(expected_data_array_x, first_element(data_array, 'x'))
    assert_identical(expected_data_array_y, first_element(data_array, 'y'))
    assert_identical(expected_dataset_x, first_element(dataset, 'x'))
    assert_identical(expected_dataset_y, first_element(dataset, 'y'))
    assert_identical(expected_data_array_x, first_element(data_array.groupby('y'), 'x'))
    assert_identical(expected_dataset_x, first_element(dataset.groupby('y'), 'x'))

    def multiply(*args):
        val = args[0]
        for arg in args[1:]:
            val = val * arg
        return val
    with pytest.raises(ValueError):
        apply_ufunc(multiply, data_array, data_array['y'].values, input_core_dims=[['y']], output_core_dims=[['y']])
    expected = xr.DataArray(multiply(data_array, data_array['y']), dims=['x', 'y'], coords=data_array.coords)
    actual = apply_ufunc(multiply, data_array, data_array['y'].values, input_core_dims=[['y'], []], output_core_dims=[['y']])
    assert_identical(expected, actual)