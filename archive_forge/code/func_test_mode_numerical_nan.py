from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('dropna, expected', [(True, [1.0]), (False, [1, np.nan])])
def test_mode_numerical_nan(self, dropna, expected):
    s = Series([1, 1, 2, np.nan, np.nan])
    result = s.mode(dropna)
    expected = Series(expected)
    tm.assert_series_equal(result, expected)