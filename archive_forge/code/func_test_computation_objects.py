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
@pytest.mark.parametrize('func', (method('groupby', 'x'), method('groupby_bins', 'x', bins=2), method('coarsen', x=2), pytest.param(method('rolling', x=3), marks=pytest.mark.xfail(reason='strips units')), pytest.param(method('rolling_exp', x=3), marks=pytest.mark.xfail(reason='numbagg functions are not supported by pint')), method('weighted', xr.DataArray(data=np.linspace(0, 1, 5), dims='y'))), ids=repr)
@pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
def test_computation_objects(self, func, variant, dtype):
    variants = {'data': ((unit_registry.degK, unit_registry.Pa), 1, 1), 'dims': ((1, 1), unit_registry.m, 1), 'coords': ((1, 1), 1, unit_registry.m)}
    (unit1, unit2), dim_unit, coord_unit = variants.get(variant)
    array1 = np.linspace(-5, 5, 4 * 5).reshape(4, 5).astype(dtype) * unit1
    array2 = np.linspace(10, 20, 4 * 3).reshape(4, 3).astype(dtype) * unit2
    x = np.arange(4) * dim_unit
    y = np.arange(5) * dim_unit
    z = np.arange(3) * dim_unit
    ds = xr.Dataset(data_vars={'a': (('x', 'y'), array1), 'b': (('x', 'z'), array2)}, coords={'x': x, 'y': y, 'z': z, 'y2': ('y', np.arange(5) * coord_unit)})
    units = extract_units(ds)
    args = [] if func.name != 'groupby' else ['y']
    kwargs = {}
    expected = attach_units(func(strip_units(ds)).mean(*args, **kwargs), units)
    actual = func(ds).mean(*args, **kwargs)
    assert_units_equal(expected, actual)
    assert_allclose(expected, actual)