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
def test_duck_array_ops(self):
    import dask.array
    d = dask.array.array([1, 2, 3])
    q = unit_registry.Quantity(d, units='m')
    da = xr.DataArray(q, dims='x')
    actual = da.mean().compute()
    actual.name = None
    expected = xr.DataArray(unit_registry.Quantity(np.array(2.0), units='m'))
    assert_units_equal(expected, actual)
    assert type(expected.data) == type(actual.data)