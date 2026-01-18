import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_cut_with_nullable_int64():
    series = Series([0, 1, 2, 3, 4, pd.NA, 6, 7], dtype='Int64')
    bins = [0, 2, 4, 6, 8]
    intervals = IntervalIndex.from_breaks(bins)
    expected = Series(Categorical.from_codes([-1, 0, 0, 1, 1, -1, 2, 3], intervals, ordered=True))
    result = cut(series, bins=bins)
    tm.assert_series_equal(result, expected)