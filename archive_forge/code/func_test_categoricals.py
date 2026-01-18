import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a_dtype', ['category', 'int64'])
@pytest.mark.parametrize('b_dtype', ['category', 'int64'])
def test_categoricals(a_dtype, b_dtype):
    g = np.random.default_rng(2)
    a = Series(g.integers(0, 3, size=100)).astype(a_dtype)
    b = Series(g.integers(0, 2, size=100)).astype(b_dtype)
    result = crosstab(a, b, margins=True, dropna=False)
    columns = Index([0, 1, 'All'], dtype='object', name='col_0')
    index = Index([0, 1, 2, 'All'], dtype='object', name='row_0')
    values = [[10, 18, 28], [23, 16, 39], [17, 16, 33], [50, 50, 100]]
    expected = DataFrame(values, index, columns)
    tm.assert_frame_equal(result, expected)
    a.loc[a == 1] = 2
    a_is_cat = isinstance(a.dtype, CategoricalDtype)
    assert not a_is_cat or a.value_counts().loc[1] == 0
    result = crosstab(a, b, margins=True, dropna=False)
    values = [[10, 18, 28], [0, 0, 0], [40, 32, 72], [50, 50, 100]]
    expected = DataFrame(values, index, columns)
    if not a_is_cat:
        expected = expected.loc[[0, 2, 'All']]
        expected['All'] = expected['All'].astype('int64')
    tm.assert_frame_equal(result, expected)