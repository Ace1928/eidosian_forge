import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('expected', [array(['2019', '2020'], dtype='datetime64[ns, UTC]'), array([0, 0], dtype='timedelta64[ns]'), array([Period('2019'), Period('2020')], dtype='period[Y-DEC]'), array([Interval(0, 1), Interval(1, 2)], dtype='interval'), array([1, np.nan], dtype='Int64')])
def test_astype_category_to_extension_dtype(self, expected):
    result = expected.astype('category').astype(expected.dtype)
    tm.assert_extension_array_equal(result, expected)