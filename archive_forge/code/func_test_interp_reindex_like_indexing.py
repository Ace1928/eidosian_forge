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
@pytest.mark.skip(reason="indexes don't support units")
@pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
@pytest.mark.parametrize('func', (method('interp_like'), method('reindex_like')), ids=repr)
def test_interp_reindex_like_indexing(self, func, unit, error, dtype):
    array1 = np.linspace(-1, 0, 10).astype(dtype)
    array2 = np.linspace(0, 1, 10).astype(dtype)
    x = np.arange(10) * unit_registry.m
    new_x = (np.arange(8) + 0.5) * unit
    ds = xr.Dataset({'a': ('x', array1), 'b': ('x', array2)}, coords={'x': x})
    units = extract_units(ds)
    other = xr.Dataset({'a': ('x', np.empty_like(new_x))}, coords={'x': new_x})
    if error is not None:
        with pytest.raises(error):
            func(ds, other)
        return
    expected = attach_units(func(strip_units(ds), other), units)
    actual = func(ds, other)
    assert_units_equal(expected, actual)
    assert_equal(expected, actual)