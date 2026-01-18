import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_string_columns(self):
    df = DataFrame([[1, 2]], columns=Index(['A', 'B'], dtype='string'))
    result = df.A
    expected = df['A']
    tm.assert_series_equal(result, expected)