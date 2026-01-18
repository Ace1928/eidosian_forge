from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multiindex_align_to_series_with_common_index_level_missing_in_right(self):
    foo_index = Index([1, 2, 3], name='foo')
    bar_index = Index([1, 2, 3, 4], name='bar')
    series = Series([1, 2], index=Index([1, 2], name='bar'), name='foo_series')
    df = DataFrame({'col': np.arange(12)}, index=pd.MultiIndex.from_product([foo_index, bar_index]))
    expected_r = Series([1, 2, np.nan, np.nan] * 3, index=df.index, name='foo_series')
    result_l, result_r = df.align(series, axis=0)
    tm.assert_frame_equal(result_l, df)
    tm.assert_series_equal(result_r, expected_r)