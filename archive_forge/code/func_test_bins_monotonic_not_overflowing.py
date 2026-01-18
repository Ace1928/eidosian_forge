import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('x, bins, expected', [(date_range('2017-12-31', periods=3), [Timestamp.min, Timestamp('2018-01-01'), Timestamp.max], IntervalIndex.from_tuples([(Timestamp.min, Timestamp('2018-01-01')), (Timestamp('2018-01-01'), Timestamp.max)])), ([-1, 0, 1], np.array([np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max], dtype='int64'), IntervalIndex.from_tuples([(np.iinfo(np.int64).min, 0), (0, np.iinfo(np.int64).max)])), ([np.timedelta64(-1, 'ns'), np.timedelta64(0, 'ns'), np.timedelta64(1, 'ns')], np.array([np.timedelta64(-np.iinfo(np.int64).max, 'ns'), np.timedelta64(0, 'ns'), np.timedelta64(np.iinfo(np.int64).max, 'ns')]), IntervalIndex.from_tuples([(np.timedelta64(-np.iinfo(np.int64).max, 'ns'), np.timedelta64(0, 'ns')), (np.timedelta64(0, 'ns'), np.timedelta64(np.iinfo(np.int64).max, 'ns'))]))])
def test_bins_monotonic_not_overflowing(x, bins, expected):
    result = cut(x, bins)
    tm.assert_index_equal(result.categories, expected)