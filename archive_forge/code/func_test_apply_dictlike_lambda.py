import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
@pytest.mark.parametrize('ops, expected', [({'a': lambda x: x}, Series([1, 2, 3], index=MultiIndex.from_arrays([['a'] * 3, range(3)]))), ({'a': lambda x: x.sum()}, Series([6], index=['a']))])
def test_apply_dictlike_lambda(ops, by_row, expected):
    ser = Series([1, 2, 3])
    result = ser.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)