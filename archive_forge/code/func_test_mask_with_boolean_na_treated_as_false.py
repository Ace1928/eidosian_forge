import math
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('index', [True, False])
def test_mask_with_boolean_na_treated_as_false(index):
    ser = Series(range(3))
    idx = Categorical([True, False, None])
    if index:
        idx = CategoricalIndex(idx)
    result = ser[idx]
    expected = ser[idx.fillna(False)]
    tm.assert_series_equal(result, expected)