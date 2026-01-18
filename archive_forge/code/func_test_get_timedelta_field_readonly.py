import numpy as np
import pytest
from pandas._libs.tslibs import fields
import pandas._testing as tm
def test_get_timedelta_field_readonly(dtindex):
    result = fields.get_timedelta_field(dtindex, 'seconds')
    expected = np.array([0] * 5, dtype=np.int32)
    tm.assert_numpy_array_equal(result, expected)