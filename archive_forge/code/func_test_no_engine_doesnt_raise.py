import pytest
from pandas import (
import pandas._testing as tm
def test_no_engine_doesnt_raise(self):
    df = DataFrame({'a': [3, 2, 3, 2], 'b': range(4), 'c': range(1, 5)})
    gb = df.groupby('a')
    with option_context('compute.use_numba', True):
        res = gb.agg({'b': 'first'})
    expected = gb.agg({'b': 'first'})
    tm.assert_frame_equal(res, expected)