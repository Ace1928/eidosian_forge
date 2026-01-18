import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_agg_apply_corner(ts, tsframe):
    grouped = ts.groupby(ts * np.nan, group_keys=False)
    assert ts.dtype == np.float64
    exp = Series([], dtype=np.float64, index=Index([], dtype=np.float64))
    tm.assert_series_equal(grouped.sum(), exp)
    tm.assert_series_equal(grouped.agg('sum'), exp)
    tm.assert_series_equal(grouped.apply('sum'), exp, check_index_type=False)
    grouped = tsframe.groupby(tsframe['A'] * np.nan, group_keys=False)
    exp_df = DataFrame(columns=tsframe.columns, dtype=float, index=Index([], name='A', dtype=np.float64))
    tm.assert_frame_equal(grouped.sum(), exp_df)
    tm.assert_frame_equal(grouped.agg('sum'), exp_df)
    msg = 'The behavior of DataFrame.sum with axis=None is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False):
        res = grouped.apply(np.sum)
    tm.assert_frame_equal(res, exp_df)