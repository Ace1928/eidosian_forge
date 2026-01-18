import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_cut_with_nonexact_categorical_indices():
    ser = Series(range(100))
    ser1 = cut(ser, 10).value_counts().head(5)
    ser2 = cut(ser, 10).value_counts().tail(5)
    result = DataFrame({'1': ser1, '2': ser2})
    index = pd.CategoricalIndex([Interval(-0.099, 9.9, closed='right'), Interval(9.9, 19.8, closed='right'), Interval(19.8, 29.7, closed='right'), Interval(29.7, 39.6, closed='right'), Interval(39.6, 49.5, closed='right'), Interval(49.5, 59.4, closed='right'), Interval(59.4, 69.3, closed='right'), Interval(69.3, 79.2, closed='right'), Interval(79.2, 89.1, closed='right'), Interval(89.1, 99, closed='right')], ordered=True)
    expected = DataFrame({'1': [10] * 5 + [np.nan] * 5, '2': [np.nan] * 5 + [10] * 5}, index=index)
    tm.assert_frame_equal(expected, result)