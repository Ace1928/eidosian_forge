import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('box', [np.array, list, tuple])
def test_crosstab_ndarray(self, box):
    a = box(np.random.default_rng(2).integers(0, 5, size=100))
    b = box(np.random.default_rng(2).integers(0, 3, size=100))
    c = box(np.random.default_rng(2).integers(0, 10, size=100))
    df = DataFrame({'a': a, 'b': b, 'c': c})
    result = crosstab(a, [b, c], rownames=['a'], colnames=('b', 'c'))
    expected = crosstab(df['a'], [df['b'], df['c']])
    tm.assert_frame_equal(result, expected)
    result = crosstab([b, c], a, colnames=['a'], rownames=('b', 'c'))
    expected = crosstab([df['b'], df['c']], df['a'])
    tm.assert_frame_equal(result, expected)
    result = crosstab(a, c)
    expected = crosstab(df['a'], df['c'])
    expected.index.names = ['row_0']
    expected.columns.names = ['col_0']
    tm.assert_frame_equal(result, expected)