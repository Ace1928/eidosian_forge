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
@pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
@pytest.mark.parametrize('func', (method('where'), method('_getitem_with_mask')), ids=repr)
def test_masking(self, func, unit, error, dtype):
    base_unit = unit_registry.m
    array = np.linspace(0, 5, 10).astype(dtype) * base_unit
    variable = xr.Variable('x', array)
    cond = np.array([True, False] * 5)
    other = -1 * unit
    if error is not None:
        with pytest.raises(error):
            func(variable, cond, other)
        return
    expected = attach_units(func(strip_units(variable), cond, strip_units(convert_units(other, {None: base_unit} if is_compatible(base_unit, unit) else {None: None}))), extract_units(variable))
    actual = func(variable, cond, other)
    assert_units_equal(expected, actual)
    assert_identical(expected, actual)