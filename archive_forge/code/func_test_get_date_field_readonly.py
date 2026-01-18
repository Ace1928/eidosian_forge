import numpy as np
import pytest
from pandas._libs.tslibs import fields
import pandas._testing as tm
def test_get_date_field_readonly(dtindex):
    result = fields.get_date_field(dtindex, 'Y')
    expected = np.array([1970, 1970, 1970, 1970, 1970], dtype=np.int32)
    tm.assert_numpy_array_equal(result, expected)