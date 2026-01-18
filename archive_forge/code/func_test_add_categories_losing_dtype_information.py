import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
def test_add_categories_losing_dtype_information(self):
    cat = Categorical(Series([1, 2], dtype='Int64'))
    ser = Series([4], dtype='Int64')
    result = cat.add_categories(ser)
    expected = Categorical(Series([1, 2], dtype='Int64'), categories=Series([1, 2, 4], dtype='Int64'))
    tm.assert_categorical_equal(result, expected)
    cat = Categorical(Series(['a', 'b', 'a'], dtype=StringDtype()))
    ser = Series(['d'], dtype=StringDtype())
    result = cat.add_categories(ser)
    expected = Categorical(Series(['a', 'b', 'a'], dtype=StringDtype()), categories=Series(['a', 'b', 'd'], dtype=StringDtype()))
    tm.assert_categorical_equal(result, expected)