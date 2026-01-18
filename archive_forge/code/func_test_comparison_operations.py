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
@pytest.mark.parametrize('comparison', (pytest.param(operator.lt, id='less_than'), pytest.param(operator.ge, id='greater_equal'), pytest.param(operator.eq, id='equal')))
@pytest.mark.parametrize('unit,error', (pytest.param(1, ValueError, id='without_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.mm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
def test_comparison_operations(self, comparison, unit, error, dtype):
    array = np.array([10.1, 5.2, 6.5, 8.0, 21.3, 7.1, 1.3]).astype(dtype) * unit_registry.m
    data_array = xr.DataArray(data=array)
    value = 8
    to_compare_with = value * unit
    if error is not None and comparison is not operator.eq:
        with pytest.raises(error):
            comparison(array, to_compare_with)
        with pytest.raises(error):
            comparison(data_array, to_compare_with)
        return
    actual = comparison(data_array, to_compare_with)
    expected_units = {None: unit_registry.m if array.check(unit) else None}
    expected = array.check(unit) & comparison(strip_units(data_array), strip_units(convert_units(to_compare_with, expected_units)))
    assert_units_equal(expected, actual)
    assert_identical(expected, actual)