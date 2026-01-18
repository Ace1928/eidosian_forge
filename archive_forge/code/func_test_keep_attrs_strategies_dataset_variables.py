from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
@pytest.mark.parametrize('variant', ('data', 'dim', 'coord'))
@pytest.mark.parametrize(['strategy', 'attrs', 'expected', 'error'], (pytest.param(None, [{'a': 1}, {'a': 2}, {'a': 3}], {}, False, id='default'), pytest.param(False, [{'a': 1}, {'a': 2}, {'a': 3}], {}, False, id='False'), pytest.param(True, [{'a': 1}, {'a': 2}, {'a': 3}], {'a': 1}, False, id='True'), pytest.param('override', [{'a': 1}, {'a': 2}, {'a': 3}], {'a': 1}, False, id='override'), pytest.param('drop', [{'a': 1}, {'a': 2}, {'a': 3}], {}, False, id='drop'), pytest.param('drop_conflicts', [{'a': 1, 'b': 2}, {'b': 1, 'c': 3}, {'c': 3, 'd': 4}], {'a': 1, 'c': 3, 'd': 4}, False, id='drop_conflicts'), pytest.param('no_conflicts', [{'a': 1}, {'b': 2}, {'b': 3}], None, True, id='no_conflicts')))
def test_keep_attrs_strategies_dataset_variables(variant, strategy, attrs, expected, error):
    compute_attrs = {'data': lambda attrs, default: (attrs, default, default), 'dim': lambda attrs, default: (default, attrs, default), 'coord': lambda attrs, default: (default, default, attrs)}.get(variant)
    data_attrs, dim_attrs, coord_attrs = compute_attrs(attrs, [{}, {}, {}])
    a = xr.Dataset({'a': ('x', [], data_attrs[0])}, coords={'x': ('x', [], dim_attrs[0]), 'u': ('x', [], coord_attrs[0])})
    b = xr.Dataset({'a': ('x', [], data_attrs[1])}, coords={'x': ('x', [], dim_attrs[1]), 'u': ('x', [], coord_attrs[1])})
    c = xr.Dataset({'a': ('x', [], data_attrs[2])}, coords={'x': ('x', [], dim_attrs[2]), 'u': ('x', [], coord_attrs[2])})
    if error:
        with pytest.raises(xr.MergeError):
            apply_ufunc(lambda *args: sum(args), a, b, c, keep_attrs=strategy)
    else:
        data_attrs, dim_attrs, coord_attrs = compute_attrs(expected, {})
        expected = xr.Dataset({'a': ('x', [], data_attrs)}, coords={'x': ('x', [], dim_attrs), 'u': ('x', [], coord_attrs)})
        actual = apply_ufunc(lambda *args: sum(args), a, b, c, keep_attrs=strategy)
        assert_identical(actual, expected)