import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_multi_column_dtype_assignment():
    df = pd.DataFrame({'a': [0.0], 'b': 0.0})
    expected = pd.DataFrame({'a': [0], 'b': 0})
    df[['a', 'b']] = 0
    tm.assert_frame_equal(df, expected)
    df['b'] = 0
    tm.assert_frame_equal(df, expected)