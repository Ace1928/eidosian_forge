import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('func', [list, np.array])
def test_bins(func):
    data = func([0.2, 1.4, 2.5, 6.2, 9.7, 2.1])
    result, bins = cut(data, 3, retbins=True)
    intervals = IntervalIndex.from_breaks(bins.round(3))
    intervals = intervals.take([0, 0, 0, 1, 2, 0])
    expected = Categorical(intervals, ordered=True)
    tm.assert_categorical_equal(result, expected)
    tm.assert_almost_equal(bins, np.array([0.1905, 3.36666667, 6.53333333, 9.7]))