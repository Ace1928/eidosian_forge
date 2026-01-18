from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_apply_columns_multilevel():
    cols = pd.MultiIndex.from_tuples([('A', 'a', '', 'one'), ('B', 'b', 'i', 'two')])
    ind = date_range(start='2017-01-01', freq='15Min', periods=8)
    df = DataFrame(np.array([0] * 16).reshape(8, 2), index=ind, columns=cols)
    agg_dict = {col: np.sum if col[3] == 'one' else np.mean for col in df.columns}
    result = df.resample('h').apply(lambda x: agg_dict[x.name](x))
    expected = DataFrame(2 * [[0, 0.0]], index=date_range(start='2017-01-01', freq='1h', periods=2), columns=pd.MultiIndex.from_tuples([('A', 'a', '', 'one'), ('B', 'b', 'i', 'two')]))
    tm.assert_frame_equal(result, expected)