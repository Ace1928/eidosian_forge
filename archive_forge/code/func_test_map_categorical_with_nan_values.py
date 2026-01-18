from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data, expected_dtype', [(['1-1', '1-1', np.nan], 'category'), (['1-1', '1-2', np.nan], object)])
def test_map_categorical_with_nan_values(data, expected_dtype, using_infer_string):

    def func(val):
        return val.split('-')[0]
    s = Series(data, dtype='category')
    result = s.map(func, na_action='ignore')
    if using_infer_string and expected_dtype == object:
        expected_dtype = 'string[pyarrow_numpy]'
    expected = Series(['1', '1', np.nan], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)