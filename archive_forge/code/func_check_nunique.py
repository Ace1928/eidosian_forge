import datetime as dt
from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def check_nunique(df, keys, as_index=True):
    original_df = df.copy()
    gr = df.groupby(keys, as_index=as_index, sort=sort)
    left = gr['julie'].nunique(dropna=dropna)
    gr = df.groupby(keys, as_index=as_index, sort=sort)
    right = gr['julie'].apply(Series.nunique, dropna=dropna)
    if not as_index:
        right = right.reset_index(drop=True)
    if as_index:
        tm.assert_series_equal(left, right, check_names=False)
    else:
        tm.assert_frame_equal(left, right, check_names=False)
    tm.assert_frame_equal(df, original_df)