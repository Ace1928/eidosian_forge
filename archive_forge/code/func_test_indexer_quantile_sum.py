import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import (
from pandas.core.indexers.objects import (
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize(('end_value', 'values'), [(1, [0.0, 1, 1, 3, 2]), (-1, [0.0, 1, 0, 3, 1])])
@pytest.mark.parametrize(('func', 'args'), [('median', []), ('quantile', [0.5])])
def test_indexer_quantile_sum(end_value, values, func, args):

    class CustomIndexer(BaseIndexer):

        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            start = np.empty(num_values, dtype=np.int64)
            end = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                if self.use_expanding[i]:
                    start[i] = 0
                    end[i] = max(i + end_value, 1)
                else:
                    start[i] = i
                    end[i] = i + self.window_size
            return (start, end)
    use_expanding = [True, False, True, False, True]
    df = DataFrame({'values': range(5)})
    indexer = CustomIndexer(window_size=1, use_expanding=use_expanding)
    result = getattr(df.rolling(indexer), func)(*args)
    expected = DataFrame({'values': values})
    tm.assert_frame_equal(result, expected)