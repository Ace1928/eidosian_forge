from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_filter_out_no_groups():
    s = Series([1, 3, 20, 5, 22, 24, 7])
    grouper = s.apply(lambda x: x % 2)
    grouped = s.groupby(grouper)
    filtered = grouped.filter(lambda x: x.mean() > 0)
    tm.assert_series_equal(filtered, s)
    df = DataFrame({'A': [1, 12, 12, 1], 'B': 'a b c d'.split()})
    grouper = df['A'].apply(lambda x: x % 2)
    grouped = df.groupby(grouper)
    filtered = grouped.filter(lambda x: x['A'].mean() > 0)
    tm.assert_frame_equal(filtered, df)