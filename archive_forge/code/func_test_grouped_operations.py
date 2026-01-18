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
@pytest.mark.parametrize('compute_backend', ['numbagg', None], indirect=True)
@pytest.mark.parametrize('func', (method('assign', c=lambda ds: 10 * ds.b), method('assign_coords', v=('x', np.arange(5) * unit_registry.s)), method('first'), method('last'), method('quantile', q=[0.25, 0.5, 0.75], dim='x')), ids=repr)
@pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
def test_grouped_operations(self, func, variant, dtype, compute_backend):
    variants = {'data': ((unit_registry.degK, unit_registry.Pa), 1, 1), 'dims': ((1, 1), unit_registry.m, 1), 'coords': ((1, 1), 1, unit_registry.m)}
    (unit1, unit2), dim_unit, coord_unit = variants.get(variant)
    array1 = np.linspace(-5, 5, 5 * 4).reshape(5, 4).astype(dtype) * unit1
    array2 = np.linspace(10, 20, 5 * 4 * 3).reshape(5, 4, 3).astype(dtype) * unit2
    x = np.arange(5) * dim_unit
    y = np.arange(4) * dim_unit
    z = np.arange(3) * dim_unit
    u = np.linspace(-1, 0, 4) * coord_unit
    ds = xr.Dataset(data_vars={'a': (('x', 'y'), array1), 'b': (('x', 'y', 'z'), array2)}, coords={'x': x, 'y': y, 'z': z, 'u': ('y', u)})
    assigned_units = {'c': unit2, 'v': unit_registry.s}
    units = merge_mappings(extract_units(ds), assigned_units)
    stripped_kwargs = {name: strip_units(value) for name, value in func.kwargs.items()}
    expected = attach_units(func(strip_units(ds).groupby('y', squeeze=False), **stripped_kwargs), units)
    actual = func(ds.groupby('y', squeeze=False))
    assert_units_equal(expected, actual)
    assert_equal(expected, actual)