import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.indexes.common import Base
@pytest.mark.parametrize('mapper', [lambda values, index: {i: e for e, i in zip(values, index)}, lambda values, index: pd.Series(values, index, dtype=object)])
def test_map_dictlike(self, mapper, simple_index):
    index = simple_index
    expected = index + index.freq
    if isinstance(expected, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        expected = expected._with_freq(None)
    result = index.map(mapper(expected, index))
    tm.assert_index_equal(result, expected)
    expected = pd.Index([pd.NaT] + index[1:].tolist())
    result = index.map(mapper(expected, index))
    tm.assert_index_equal(result, expected)
    expected = pd.Index([np.nan] * len(index))
    result = index.map(mapper([], []))
    tm.assert_index_equal(result, expected)