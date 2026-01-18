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
@pytest.mark.parametrize('func', (function(lambda x, *_: +x, function_label='unary_plus'), function(lambda x, *_: -x, function_label='unary_minus'), function(lambda x, *_: abs(x), function_label='absolute'), function(lambda x, y: x + y, function_label='sum'), function(lambda x, y: y + x, function_label='commutative_sum'), function(lambda x, y: x * y, function_label='product'), function(lambda x, y: y * x, function_label='commutative_product')), ids=repr)
def test_1d_math(self, func, unit, error, dtype):
    base_unit = unit_registry.m
    array = np.arange(5).astype(dtype) * base_unit
    variable = xr.Variable('x', array)
    values = np.ones(5)
    y = values * unit
    if error is not None and func.name in ('sum', 'commutative_sum'):
        with pytest.raises(error):
            func(variable, y)
        return
    units = extract_units(func(array, y))
    if all(compatible_mappings(units, extract_units(y)).values()):
        converted_y = convert_units(y, units)
    else:
        converted_y = y
    if all(compatible_mappings(units, extract_units(variable)).values()):
        converted_variable = convert_units(variable, units)
    else:
        converted_variable = variable
    expected = attach_units(func(strip_units(converted_variable), strip_units(converted_y)), units)
    actual = func(variable, y)
    assert_units_equal(expected, actual)
    assert_allclose(expected, actual)