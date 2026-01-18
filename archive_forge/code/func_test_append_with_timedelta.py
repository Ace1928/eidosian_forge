import datetime
from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
def test_append_with_timedelta(setup_path):
    ts = Timestamp('20130101').as_unit('ns')
    df = DataFrame({'A': ts, 'B': [ts + timedelta(days=i, seconds=10) for i in range(10)]})
    df['C'] = df['A'] - df['B']
    df.loc[3:5, 'C'] = np.nan
    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, 'df')
        store.append('df', df, data_columns=True)
        result = store.select('df')
        tm.assert_frame_equal(result, df)
        result = store.select('df', where='C<100000')
        tm.assert_frame_equal(result, df)
        result = store.select('df', where="C<pd.Timedelta('-3D')")
        tm.assert_frame_equal(result, df.iloc[3:])
        result = store.select('df', "C<'-3D'")
        tm.assert_frame_equal(result, df.iloc[3:])
        result = store.select('df', "C<'-500000s'")
        result = result.dropna(subset=['C'])
        tm.assert_frame_equal(result, df.iloc[6:])
        result = store.select('df', "C<'-3.5D'")
        result = result.iloc[1:]
        tm.assert_frame_equal(result, df.iloc[4:])
        _maybe_remove(store, 'df2')
        store.put('df2', df)
        result = store.select('df2')
        tm.assert_frame_equal(result, df)