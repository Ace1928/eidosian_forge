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
def test_broadcast_dataset(dtype):
    array1 = np.linspace(0, 10, 2) * unit_registry.Pa
    array2 = np.linspace(0, 10, 3) * unit_registry.Pa
    x1 = np.arange(2)
    y1 = np.arange(3)
    x2 = np.arange(2, 4)
    y2 = np.arange(3, 6)
    ds = xr.Dataset(data_vars={'a': ('x', array1), 'b': ('y', array2)}, coords={'x': x1, 'y': y1})
    other = xr.Dataset(data_vars={'a': ('x', array1.to(unit_registry.hPa)), 'b': ('y', array2.to(unit_registry.hPa))}, coords={'x': x2, 'y': y2})
    units_a = extract_units(ds)
    units_b = extract_units(other)
    expected_a, expected_b = xr.broadcast(strip_units(ds), strip_units(other))
    expected_a = attach_units(expected_a, units_a)
    expected_b = attach_units(expected_b, units_b)
    actual_a, actual_b = xr.broadcast(ds, other)
    assert_units_equal(expected_a, actual_a)
    assert_identical(expected_a, actual_a)
    assert_units_equal(expected_b, actual_b)
    assert_identical(expected_b, actual_b)