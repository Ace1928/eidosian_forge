import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_no_right():
    data = np.array([0.2, 1.4, 2.5, 6.2, 9.7, 2.1, 2.575])
    result, bins = cut(data, 4, right=False, retbins=True)
    intervals = IntervalIndex.from_breaks(bins.round(3), closed='left')
    intervals = intervals.take([0, 0, 0, 2, 3, 0, 1])
    expected = Categorical(intervals, ordered=True)
    tm.assert_categorical_equal(result, expected)
    tm.assert_almost_equal(bins, np.array([0.2, 2.575, 4.95, 7.325, 9.7095]))