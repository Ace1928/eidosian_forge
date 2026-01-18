import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
def test_custom_asserts():
    data = JSONArray([collections.UserDict({'a': 1}), collections.UserDict({'b': 2}), collections.UserDict({'c': 3})])
    a = pd.Series(data)
    custom_assert_series_equal(a, a)
    custom_assert_frame_equal(a.to_frame(), a.to_frame())
    b = pd.Series(data.take([0, 0, 1]))
    msg = 'Series are different'
    with pytest.raises(AssertionError, match=msg):
        custom_assert_series_equal(a, b)
    with pytest.raises(AssertionError, match=msg):
        custom_assert_frame_equal(a.to_frame(), b.to_frame())