import numpy as np
from pandas import (
import pandas._testing as tm
def test_categorical_nan_equality(self):
    cat = Series(Categorical(['a', 'b', 'c', np.nan]))
    expected = Series([True, True, True, False])
    result = cat == cat
    tm.assert_series_equal(result, expected)