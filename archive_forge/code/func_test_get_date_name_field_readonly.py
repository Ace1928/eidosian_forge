import numpy as np
import pytest
from pandas._libs.tslibs import fields
import pandas._testing as tm
def test_get_date_name_field_readonly(dtindex):
    result = fields.get_date_name_field(dtindex, 'month_name')
    expected = np.array(['January', 'February', 'March', 'April', 'May'], dtype=object)
    tm.assert_numpy_array_equal(result, expected)