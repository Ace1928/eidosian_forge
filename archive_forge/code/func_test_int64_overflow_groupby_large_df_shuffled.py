from collections import defaultdict
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
import pandas.core.common as com
from pandas.core.sorting import (
@pytest.mark.parametrize('agg', ['mean', 'median'])
def test_int64_overflow_groupby_large_df_shuffled(self, agg):
    rs = np.random.default_rng(2)
    arr = rs.integers(-1 << 12, 1 << 12, (1 << 15, 5))
    i = rs.choice(len(arr), len(arr) * 4)
    arr = np.vstack((arr, arr[i]))
    i = rs.permutation(len(arr))
    arr = arr[i]
    df = DataFrame(arr, columns=list('abcde'))
    df['jim'], df['joe'] = np.zeros((2, len(df)))
    gr = df.groupby(list('abcde'))
    assert is_int64_overflow_possible(gr._grouper.shape)
    mi = MultiIndex.from_arrays([ar.ravel() for ar in np.array_split(np.unique(arr, axis=0), 5, axis=1)], names=list('abcde'))
    res = DataFrame(np.zeros((len(mi), 2)), columns=['jim', 'joe'], index=mi).sort_index()
    tm.assert_frame_equal(getattr(gr, agg)(), res)