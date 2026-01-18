import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_no_prefix_string_cats_basic_bool_values():
    dummies = DataFrame({'a': [True, False, False, True], 'b': [False, True, False, False], 'c': [False, False, True, False]})
    expected = DataFrame({'': ['a', 'b', 'c', 'a']})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)