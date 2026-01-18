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
def test_append_to_multiple_dropna(setup_path):
    df1 = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    df2 = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B')).rename(columns='{}_2'.format)
    df1.iloc[1, df1.columns.get_indexer(['A', 'B'])] = np.nan
    df = concat([df1, df2], axis=1)
    with ensure_clean_store(setup_path) as store:
        store.append_to_multiple({'df1': ['A', 'B'], 'df2': None}, df, selector='df1', dropna=True)
        result = store.select_as_multiple(['df1', 'df2'])
        expected = df.dropna()
        tm.assert_frame_equal(result, expected, check_index_type=True)
        tm.assert_index_equal(store.select('df1').index, store.select('df2').index)