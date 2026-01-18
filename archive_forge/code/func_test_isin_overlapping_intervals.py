import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas._libs.arrays import NDArrayBacked
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_isin_overlapping_intervals(self):
    idx = pd.IntervalIndex([pd.Interval(0, 2), pd.Interval(0, 1)])
    result = CategoricalIndex(idx).isin(idx)
    expected = np.array([True, True])
    tm.assert_numpy_array_equal(result, expected)