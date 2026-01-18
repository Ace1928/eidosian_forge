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
@pytest.mark.parametrize('func', (method('head', x=7, y=3, z=6), method('tail', x=7, y=3, z=6), method('thin', x=7, y=3, z=6)), ids=repr)
@pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
def test_head_tail_thin(self, func, variant, dtype):
    variants = {'data': ((unit_registry.degK, unit_registry.Pa), 1, 1), 'dims': ((1, 1), unit_registry.m, 1), 'coords': ((1, 1), 1, unit_registry.m)}
    (unit_a, unit_b), dim_unit, coord_unit = variants.get(variant)
    array1 = np.linspace(1, 2, 10 * 5).reshape(10, 5) * unit_a
    array2 = np.linspace(1, 2, 10 * 8).reshape(10, 8) * unit_b
    coords = {'x': np.arange(10) * dim_unit, 'y': np.arange(5) * dim_unit, 'z': np.arange(8) * dim_unit, 'u': ('x', np.linspace(0, 1, 10) * coord_unit), 'v': ('y', np.linspace(1, 2, 5) * coord_unit), 'w': ('z', np.linspace(-1, 0, 8) * coord_unit)}
    ds = xr.Dataset(data_vars={'a': xr.DataArray(data=array1, dims=('x', 'y')), 'b': xr.DataArray(data=array2, dims=('x', 'z'))}, coords=coords)
    expected = attach_units(func(strip_units(ds)), extract_units(ds))
    actual = func(ds)
    assert_units_equal(expected, actual)
    assert_equal(expected, actual)