from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
@pytest.mark.parametrize('method, expected', [('isnumeric', [False, True, True, False, True, True, False]), ('isdecimal', [False, True, False, False, False, True, False])])
def test_isnumeric_unicode(method, expected, any_string_dtype):
    ser = Series(['A', '3', '¼', '★', '፸', '３', 'four'], dtype=any_string_dtype)
    expected_dtype = 'bool' if any_string_dtype in object_pyarrow_numpy else 'boolean'
    expected = Series(expected, dtype=expected_dtype)
    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)
    expected = [getattr(item, method)() for item in ser]
    assert list(result) == expected