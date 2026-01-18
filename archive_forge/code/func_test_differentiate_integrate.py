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
@pytest.mark.parametrize('variant', (pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
@pytest.mark.parametrize('func', (method('differentiate', fallback_func=np.gradient), method('integrate', fallback_func=duck_array_ops.cumulative_trapezoid), method('cumulative_integrate', fallback_func=duck_array_ops.trapz)), ids=repr)
def test_differentiate_integrate(self, func, variant, dtype):
    data_unit = unit_registry.m
    unit = unit_registry.s
    variants = {'dims': ('x', unit, 1), 'coords': ('u', 1, unit)}
    coord, dim_unit, coord_unit = variants.get(variant)
    array = np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * data_unit
    x = np.arange(array.shape[0]) * dim_unit
    y = np.arange(array.shape[1]) * dim_unit
    u = np.linspace(0, 1, array.shape[0]) * coord_unit
    data_array = xr.DataArray(data=array, coords={'x': x, 'y': y, 'u': ('x', u)}, dims=('x', 'y'))
    units = extract_units(data_array)
    units.update(extract_units(func(data_array.data, getattr(data_array, coord).data, axis=0)))
    expected = attach_units(func(strip_units(data_array), coord=strip_units(coord)), units)
    actual = func(data_array, coord=coord)
    assert_units_equal(expected, actual)
    assert_identical(expected, actual)