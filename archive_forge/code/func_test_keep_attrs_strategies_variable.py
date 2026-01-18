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
@pytest.mark.parametrize(['strategy', 'attrs', 'expected', 'error'], (pytest.param(None, [{'a': 1}, {'a': 2}, {'a': 3}], {}, False, id='default'), pytest.param(False, [{'a': 1}, {'a': 2}, {'a': 3}], {}, False, id='False'), pytest.param(True, [{'a': 1}, {'a': 2}, {'a': 3}], {'a': 1}, False, id='True'), pytest.param('override', [{'a': 1}, {'a': 2}, {'a': 3}], {'a': 1}, False, id='override'), pytest.param('drop', [{'a': 1}, {'a': 2}, {'a': 3}], {}, False, id='drop'), pytest.param('drop_conflicts', [{'a': 1, 'b': 2}, {'b': 1, 'c': 3}, {'c': 3, 'd': 4}], {'a': 1, 'c': 3, 'd': 4}, False, id='drop_conflicts'), pytest.param('no_conflicts', [{'a': 1}, {'b': 2}, {'b': 3}], None, True, id='no_conflicts')))
def test_keep_attrs_strategies_variable(strategy, attrs, expected, error) -> None:
    a = xr.Variable('x', [0, 1], attrs=attrs[0])
    b = xr.Variable('x', [0, 1], attrs=attrs[1])
    c = xr.Variable('x', [0, 1], attrs=attrs[2])
    if error:
        with pytest.raises(xr.MergeError):
            apply_ufunc(lambda *args: sum(args), a, b, c, keep_attrs=strategy)
    else:
        expected = xr.Variable('x', [0, 3], attrs=expected)
        actual = apply_ufunc(lambda *args: sum(args), a, b, c, keep_attrs=strategy)
        assert_identical(actual, expected)