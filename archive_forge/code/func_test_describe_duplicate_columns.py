import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
def test_describe_duplicate_columns():
    df = DataFrame([[0, 1, 2, 3]])
    df.columns = [0, 1, 2, 0]
    gb = df.groupby(df[1])
    result = gb.describe(percentiles=[])
    columns = ['count', 'mean', 'std', 'min', '50%', 'max']
    frames = [DataFrame([[1.0, val, np.nan, val, val, val]], index=[1], columns=columns) for val in (0.0, 2.0, 3.0)]
    expected = pd.concat(frames, axis=1)
    expected.columns = MultiIndex(levels=[[0, 2], columns], codes=[6 * [0] + 6 * [1] + 6 * [0], 3 * list(range(6))])
    expected.index.names = [1]
    tm.assert_frame_equal(result, expected)