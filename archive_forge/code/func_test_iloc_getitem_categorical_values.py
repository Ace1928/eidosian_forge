from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_iloc_getitem_categorical_values(self):
    ser = Series([1, 2, 3]).astype('category')
    result = ser.iloc[0:2]
    expected = Series([1, 2]).astype(CategoricalDtype([1, 2, 3]))
    tm.assert_series_equal(result, expected)
    result = ser.iloc[[0, 1]]
    expected = Series([1, 2]).astype(CategoricalDtype([1, 2, 3]))
    tm.assert_series_equal(result, expected)
    result = ser.iloc[[True, False, False]]
    expected = Series([1]).astype(CategoricalDtype([1, 2, 3]))
    tm.assert_series_equal(result, expected)