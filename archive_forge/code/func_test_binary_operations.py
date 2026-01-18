from __future__ import annotations
import functools
import operator
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, duck_array_ops
from xarray.tests import (
from xarray.tests.test_plot import PlotTestCase
from xarray.tests.test_variable import _PAD_XR_NP_ARGS
@pytest.mark.parametrize('func', (pytest.param(lambda x: 2 * x, id='multiply'), pytest.param(lambda x: x + x, id='add'), pytest.param(lambda x: x[0] + x, id='add scalar'), pytest.param(lambda x: x.T @ x, id='matrix multiply')))
def test_binary_operations(self, func, dtype):
    array = np.arange(10).astype(dtype) * unit_registry.m
    data_array = xr.DataArray(data=array)
    units = extract_units(func(array))
    with xr.set_options(use_opt_einsum=False):
        expected = attach_units(func(strip_units(data_array)), units)
        actual = func(data_array)
    assert_units_equal(expected, actual)
    assert_identical(expected, actual)