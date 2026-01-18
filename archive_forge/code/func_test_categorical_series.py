from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize(['series', 'data'], [(Series(range(4)), {'A': [0, 3], 'B': [1, 2]}), (Series(range(4)).rename(lambda idx: idx + 1), {'A': [2], 'B': [0, 1]}), (Series(range(7)), {'A': [0, 3], 'B': [1, 2]})])
def test_categorical_series(series, data):
    groupby = series.groupby(Series(list('ABBA'), dtype='category'), observed=False)
    result = groupby.aggregate(list)
    expected = Series(data, index=CategoricalIndex(data.keys()))
    tm.assert_series_equal(result, expected)