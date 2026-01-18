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
@pytest.mark.parametrize('variant', ('data', 'coords'))
@pytest.mark.parametrize('func', (pytest.param(method('interp_like'), marks=pytest.mark.xfail(reason='uses scipy')), method('reindex_like')), ids=repr)
def test_interp_reindex_like(self, func, variant, dtype):
    variants = {'data': (unit_registry.m, 1), 'coords': (1, unit_registry.m)}
    data_unit, coord_unit = variants.get(variant)
    array1 = np.linspace(-1, 0, 10).astype(dtype) * data_unit
    array2 = np.linspace(0, 1, 10).astype(dtype) * data_unit
    y = np.arange(10) * coord_unit
    x = np.arange(10)
    new_x = np.arange(8) + 0.5
    ds = xr.Dataset({'a': ('x', array1), 'b': ('x', array2)}, coords={'x': x, 'y': ('x', y)})
    units = extract_units(ds)
    other = xr.Dataset({'a': ('x', np.empty_like(new_x))}, coords={'x': new_x})
    expected = attach_units(func(strip_units(ds), other), units)
    actual = func(ds, other)
    assert_units_equal(expected, actual)
    assert_equal(expected, actual)