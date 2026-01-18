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
def test_apply_two_inputs() -> None:
    array = np.array([1, 2, 3])
    variable = xr.Variable('x', array)
    data_array = xr.DataArray(variable, [('x', -array)])
    dataset = xr.Dataset({'y': variable}, {'x': -array})
    zero_array = np.zeros_like(array)
    zero_variable = xr.Variable('x', zero_array)
    zero_data_array = xr.DataArray(zero_variable, [('x', -array)])
    zero_dataset = xr.Dataset({'y': zero_variable}, {'x': -array})
    assert_identical(array, add(array, zero_array))
    assert_identical(array, add(zero_array, array))
    assert_identical(variable, add(variable, zero_array))
    assert_identical(variable, add(variable, zero_variable))
    assert_identical(variable, add(zero_array, variable))
    assert_identical(variable, add(zero_variable, variable))
    assert_identical(data_array, add(data_array, zero_array))
    assert_identical(data_array, add(data_array, zero_variable))
    assert_identical(data_array, add(data_array, zero_data_array))
    assert_identical(data_array, add(zero_array, data_array))
    assert_identical(data_array, add(zero_variable, data_array))
    assert_identical(data_array, add(zero_data_array, data_array))
    assert_identical(dataset, add(dataset, zero_array))
    assert_identical(dataset, add(dataset, zero_variable))
    assert_identical(dataset, add(dataset, zero_data_array))
    assert_identical(dataset, add(dataset, zero_dataset))
    assert_identical(dataset, add(zero_array, dataset))
    assert_identical(dataset, add(zero_variable, dataset))
    assert_identical(dataset, add(zero_data_array, dataset))
    assert_identical(dataset, add(zero_dataset, dataset))
    assert_identical(data_array, add(data_array.groupby('x'), zero_data_array))
    assert_identical(data_array, add(zero_data_array, data_array.groupby('x')))
    assert_identical(dataset, add(data_array.groupby('x'), zero_dataset))
    assert_identical(dataset, add(zero_dataset, data_array.groupby('x')))
    assert_identical(dataset, add(dataset.groupby('x'), zero_data_array))
    assert_identical(dataset, add(dataset.groupby('x'), zero_dataset))
    assert_identical(dataset, add(zero_data_array, dataset.groupby('x')))
    assert_identical(dataset, add(zero_dataset, dataset.groupby('x')))