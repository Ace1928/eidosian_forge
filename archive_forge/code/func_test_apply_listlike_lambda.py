import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
@pytest.mark.parametrize('ops, expected', [([lambda x: x], DataFrame({'<lambda>': [1, 2, 3]})), ([lambda x: x.sum()], Series([6], index=['<lambda>']))])
def test_apply_listlike_lambda(ops, expected, by_row):
    ser = Series([1, 2, 3])
    result = ser.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)