from __future__ import annotations
import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest
from dask.dataframe.utils import get_string_dtype
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_150
def test_inplace_modification_read_only():
    arr = np.array([(1, 2), None, 1], dtype='object')
    base = pd.Series(arr, copy=False, dtype=object, name='a')
    base_copy = pickle.loads(pickle.dumps(base))
    base_copy.values.flags.writeable = False
    dtype = get_string_dtype()
    tm.assert_series_equal(dd.from_array(base_copy.values, columns='a').compute(), base.astype(dtype))