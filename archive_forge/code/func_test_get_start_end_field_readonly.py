import numpy as np
import pytest
from pandas._libs.tslibs import fields
import pandas._testing as tm
def test_get_start_end_field_readonly(dtindex):
    result = fields.get_start_end_field(dtindex, 'is_month_start', None)
    expected = np.array([True, False, False, False, False], dtype=np.bool_)
    tm.assert_numpy_array_equal(result, expected)