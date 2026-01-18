import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_groupby_nth_interval():
    idx_result = MultiIndex([pd.CategoricalIndex([pd.Interval(0, 1), pd.Interval(1, 2)]), pd.CategoricalIndex([pd.Interval(0, 10), pd.Interval(10, 20)])], [[0, 0, 0, 1, 1], [0, 1, 1, 0, -1]])
    df_result = DataFrame({'col': range(len(idx_result))}, index=idx_result)
    result = df_result.groupby(level=[0, 1], observed=False).nth(0)
    val_expected = [0, 1, 3]
    idx_expected = MultiIndex([pd.CategoricalIndex([pd.Interval(0, 1), pd.Interval(1, 2)]), pd.CategoricalIndex([pd.Interval(0, 10), pd.Interval(10, 20)])], [[0, 0, 1], [0, 1, 0]])
    expected = DataFrame(val_expected, index=idx_expected, columns=['col'])
    tm.assert_frame_equal(result, expected)