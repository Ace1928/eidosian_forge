import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import (
from pandas.core.indexers.objects import (
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('indexer_class', [FixedWindowIndexer, FixedForwardWindowIndexer, ExpandingIndexer])
@pytest.mark.parametrize('window_size', [1, 2, 12])
@pytest.mark.parametrize('df_data', [{'a': [1, 1], 'b': [0, 1]}, {'a': [1, 2], 'b': [0, 1]}, {'a': [1] * 16, 'b': [np.nan, 1, 2, np.nan] + list(range(4, 16))}])
def test_indexers_are_reusable_after_groupby_rolling(indexer_class, window_size, df_data):
    df = DataFrame(df_data)
    num_trials = 3
    indexer = indexer_class(window_size=window_size)
    original_window_size = indexer.window_size
    for i in range(num_trials):
        df.groupby('a')['b'].rolling(window=indexer, min_periods=1).mean()
        assert indexer.window_size == original_window_size