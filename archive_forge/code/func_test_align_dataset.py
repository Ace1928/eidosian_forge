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
@pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.mm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')), ids=repr)
@pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
@pytest.mark.parametrize('value', (10, dtypes.NA))
def test_align_dataset(value, unit, variant, error, dtype):
    if variant == 'coords' and (value != dtypes.NA or isinstance(unit, unit_registry.Unit)):
        pytest.xfail(reason='fill_value is used for both data variables and coords. See https://github.com/pydata/xarray/issues/4165')
    fill_value = dtypes.get_fill_value(dtype) if value == dtypes.NA else value
    original_unit = unit_registry.m
    variants = {'data': ((original_unit, unit), (1, 1), (1, 1)), 'dims': ((1, 1), (original_unit, unit), (1, 1)), 'coords': ((1, 1), (1, 1), (original_unit, unit))}
    (data_unit1, data_unit2), (dim_unit1, dim_unit2), (coord_unit1, coord_unit2) = variants.get(variant)
    array1 = np.linspace(0, 10, 2 * 5).reshape(2, 5).astype(dtype) * data_unit1
    array2 = np.linspace(0, 10, 2 * 5).reshape(2, 5).astype(dtype) * data_unit2
    x = np.arange(2) * dim_unit1
    y1 = np.arange(5) * dim_unit1
    y2 = np.arange(2, 7) * dim_unit2
    u1 = np.array([3, 5, 7, 8, 9]) * coord_unit1
    u2 = np.array([7, 8, 9, 11, 13]) * coord_unit2
    coords1 = {'x': x, 'y': y1}
    coords2 = {'x': x, 'y': y2}
    if variant == 'coords':
        coords1['u'] = ('y', u1)
        coords2['u'] = ('y', u2)
    ds1 = xr.Dataset(data_vars={'a': (('x', 'y'), array1)}, coords=coords1)
    ds2 = xr.Dataset(data_vars={'a': (('x', 'y'), array2)}, coords=coords2)
    fill_value = fill_value * data_unit2
    func = function(xr.align, join='outer', fill_value=fill_value)
    if error is not None and (value != dtypes.NA or isinstance(fill_value, Quantity)):
        with pytest.raises(error):
            func(ds1, ds2)
        return
    stripped_kwargs = {key: strip_units(convert_units(value, {None: data_unit1 if data_unit2 != 1 else None})) for key, value in func.kwargs.items()}
    units_a = extract_units(ds1)
    units_b = extract_units(ds2)
    expected_a, expected_b = func(strip_units(ds1), strip_units(convert_units(ds2, units_a)), **stripped_kwargs)
    expected_a = attach_units(expected_a, units_a)
    if isinstance(array2, Quantity):
        expected_b = convert_units(attach_units(expected_b, units_a), units_b)
    else:
        expected_b = attach_units(expected_b, units_b)
    actual_a, actual_b = func(ds1, ds2)
    assert_units_equal(expected_a, actual_a)
    assert_allclose(expected_a, actual_a)
    assert_units_equal(expected_b, actual_b)
    assert_allclose(expected_b, actual_b)