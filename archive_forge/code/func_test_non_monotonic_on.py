import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_non_monotonic_on(self):
    df = DataFrame({'A': date_range('20130101', periods=5, freq='s'), 'B': range(5)})
    df = df.set_index('A')
    non_monotonic_index = df.index.to_list()
    non_monotonic_index[0] = non_monotonic_index[3]
    df.index = non_monotonic_index
    assert not df.index.is_monotonic_increasing
    msg = 'index values must be monotonic'
    with pytest.raises(ValueError, match=msg):
        df.rolling('2s').sum()
    df = df.reset_index()
    msg = 'invalid on specified as A, must be a column \\(of DataFrame\\), an Index or None'
    with pytest.raises(ValueError, match=msg):
        df.rolling('2s', on='A').sum()