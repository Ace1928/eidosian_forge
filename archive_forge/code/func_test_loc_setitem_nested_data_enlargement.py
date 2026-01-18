from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_loc_setitem_nested_data_enlargement():
    df = DataFrame({'a': [1]})
    ser = Series({'label': df})
    ser.loc['new_label'] = df
    expected = Series({'label': df, 'new_label': df})
    tm.assert_series_equal(ser, expected)