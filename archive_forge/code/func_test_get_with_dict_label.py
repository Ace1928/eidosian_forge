from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_get_with_dict_label():
    s = Series([{'name': 'Hello', 'value': 'World'}, {'name': 'Goodbye', 'value': 'Planet'}, {'value': 'Sea'}])
    result = s.str.get('name')
    expected = Series(['Hello', 'Goodbye', None], dtype=object)
    tm.assert_series_equal(result, expected)
    result = s.str.get('value')
    expected = Series(['World', 'Planet', 'Sea'], dtype=object)
    tm.assert_series_equal(result, expected)