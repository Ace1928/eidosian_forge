from datetime import datetime
import numpy as np
from pandas import (
import pandas._testing as tm
def test_multiindex_period_datetime():
    idx1 = Index(['a', 'a', 'a', 'b', 'b'])
    idx2 = period_range('2012-01', periods=len(idx1), freq='M')
    s = Series(np.random.default_rng(2).standard_normal(len(idx1)), [idx1, idx2])
    expected = s.iloc[0]
    result = s.loc['a', Period('2012-01')]
    assert result == expected
    result = s.loc['a', datetime(2012, 1, 1)]
    assert result == expected