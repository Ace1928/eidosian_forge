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
def test_append_frame_column_oriented(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        df.index = df.index._with_freq(None)
        _maybe_remove(store, 'df1')
        store.append('df1', df.iloc[:, :2], axes=['columns'])
        store.append('df1', df.iloc[:, 2:])
        tm.assert_frame_equal(store['df1'], df)
        result = store.select('df1', 'columns=A')
        expected = df.reindex(columns=['A'])
        tm.assert_frame_equal(expected, result)
        result = store.select('df1', ('columns=A', 'index=df.index[0:4]'))
        expected = df.reindex(columns=['A'], index=df.index[0:4])
        tm.assert_frame_equal(expected, result)
        msg = re.escape('passing a filterable condition to a non-table indexer [Filter: Not Initialized]')
        with pytest.raises(TypeError, match=msg):
            store.select('df1', 'columns=A and index>df.index[4]')