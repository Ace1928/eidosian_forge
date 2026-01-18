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
def test_apply_groupby_add() -> None:
    array = np.arange(5)
    variable = xr.Variable('x', array)
    coords = {'x': -array, 'y': ('x', [0, 0, 1, 1, 2])}
    data_array = xr.DataArray(variable, coords, dims='x')
    dataset = xr.Dataset({'z': variable}, coords)
    other_variable = xr.Variable('y', [0, 10])
    other_data_array = xr.DataArray(other_variable, dims='y')
    other_dataset = xr.Dataset({'z': other_variable})
    expected_variable = xr.Variable('x', [0, 1, 12, 13, np.nan])
    expected_data_array = xr.DataArray(expected_variable, coords, dims='x')
    expected_dataset = xr.Dataset({'z': expected_variable}, coords)
    assert_identical(expected_data_array, add(data_array.groupby('y'), other_data_array))
    assert_identical(expected_dataset, add(data_array.groupby('y'), other_dataset))
    assert_identical(expected_dataset, add(dataset.groupby('y'), other_data_array))
    assert_identical(expected_dataset, add(dataset.groupby('y'), other_dataset))
    with pytest.raises(ValueError):
        add(data_array.groupby('y'), other_variable)
    with pytest.raises(ValueError):
        add(data_array.groupby('y'), data_array[:4].groupby('y'))
    with pytest.raises(ValueError):
        add(data_array.groupby('y'), data_array[1:].groupby('y'))
    with pytest.raises(ValueError):
        add(data_array.groupby('y'), other_data_array.groupby('y'))
    with pytest.raises(ValueError):
        add(data_array.groupby('y'), data_array.groupby('x'))