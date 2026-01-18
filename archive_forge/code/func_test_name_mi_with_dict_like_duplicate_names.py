import pytest
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm
@pytest.mark.parametrize('func', ['rename', 'set_names'])
@pytest.mark.parametrize('rename_dict, exp_names', [({'x': 'z'}, ['z', 'y', 'z']), ({'x': 'z', 'y': 'x'}, ['z', 'x', 'z']), ({'y': 'z'}, ['x', 'z', 'x']), ({}, ['x', 'y', 'x']), ({'z': 'a'}, ['x', 'y', 'x']), ({'y': 'z', 'a': 'b'}, ['x', 'z', 'x'])])
def test_name_mi_with_dict_like_duplicate_names(func, rename_dict, exp_names):
    mi = MultiIndex.from_arrays([[1, 2], [3, 4], [5, 6]], names=['x', 'y', 'x'])
    result = getattr(mi, func)(rename_dict)
    expected = MultiIndex.from_arrays([[1, 2], [3, 4], [5, 6]], names=exp_names)
    tm.assert_index_equal(result, expected)