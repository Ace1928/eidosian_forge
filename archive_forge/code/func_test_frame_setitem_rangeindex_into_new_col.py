from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
def test_frame_setitem_rangeindex_into_new_col(self):
    df = DataFrame({'a': ['a', 'b']})
    df['b'] = df.index
    df.loc[[False, True], 'b'] = 100
    result = df.loc[[1], :]
    expected = DataFrame({'a': ['b'], 'b': [100]}, index=[1])
    tm.assert_frame_equal(result, expected)