import datetime as dt
from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_empty_categorical(observed):
    cat = Series([1]).astype('category')
    ser = cat[:0]
    gb = ser.groupby(ser, observed=observed)
    result = gb.nunique()
    if observed:
        expected = Series([], index=cat[:0], dtype='int64')
    else:
        expected = Series([0], index=cat, dtype='int64')
    tm.assert_series_equal(result, expected)