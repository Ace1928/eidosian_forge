import pytest
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm
@pytest.mark.parametrize('func', ['rename', 'set_names'])
@pytest.mark.parametrize('rename_dict, exp_names', [({'x': 'z'}, ['z', 'y']), ({'x': 'z', 'y': 'x'}, ['z', 'x']), ({'a': 'z'}, ['x', 'y']), ({}, ['x', 'y'])])
def test_name_mi_with_dict_like(func, rename_dict, exp_names):
    mi = MultiIndex.from_arrays([[1, 2], [3, 4]], names=['x', 'y'])
    result = getattr(mi, func)(rename_dict)
    expected = MultiIndex.from_arrays([[1, 2], [3, 4]], names=exp_names)
    tm.assert_index_equal(result, expected)