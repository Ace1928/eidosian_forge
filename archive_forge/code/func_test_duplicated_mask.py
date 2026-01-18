import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('keep, vals', [('last', [True, True, False]), ('first', [False, True, True]), (False, [True, True, True])])
def test_duplicated_mask(keep, vals):
    ser = Series([1, 2, NA, NA, NA], dtype='Int64')
    result = ser.duplicated(keep=keep)
    expected = Series([False, False] + vals)
    tm.assert_series_equal(result, expected)