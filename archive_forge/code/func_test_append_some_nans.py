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
def test_append_some_nans(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame({'A': Series(np.random.default_rng(2).standard_normal(20)).astype('int32'), 'A1': np.random.default_rng(2).standard_normal(20), 'A2': np.random.default_rng(2).standard_normal(20), 'B': 'foo', 'C': 'bar', 'D': Timestamp('2001-01-01').as_unit('ns'), 'E': Timestamp('2001-01-02').as_unit('ns')}, index=np.arange(20))
        _maybe_remove(store, 'df1')
        df.loc[0:15, ['A1', 'B', 'D', 'E']] = np.nan
        store.append('df1', df[:10])
        store.append('df1', df[10:])
        tm.assert_frame_equal(store['df1'], df, check_index_type=True)
        df1 = df.copy()
        df1['A1'] = np.nan
        _maybe_remove(store, 'df1')
        store.append('df1', df1[:10])
        store.append('df1', df1[10:])
        tm.assert_frame_equal(store['df1'], df1, check_index_type=True)
        df2 = df.copy()
        df2['A2'] = np.nan
        _maybe_remove(store, 'df2')
        store.append('df2', df2[:10])
        store.append('df2', df2[10:])
        tm.assert_frame_equal(store['df2'], df2, check_index_type=True)
        df3 = df.copy()
        df3['E'] = np.nan
        _maybe_remove(store, 'df3')
        store.append('df3', df3[:10])
        store.append('df3', df3[10:])
        tm.assert_frame_equal(store['df3'], df3, check_index_type=True)