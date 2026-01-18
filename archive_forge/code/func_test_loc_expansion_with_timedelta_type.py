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
def test_loc_expansion_with_timedelta_type(self):
    result = DataFrame(columns=list('abc'))
    result.loc[0] = {'a': pd.to_timedelta(5, unit='s'), 'b': pd.to_timedelta(72, unit='s'), 'c': '23'}
    expected = DataFrame([[pd.Timedelta('0 days 00:00:05'), pd.Timedelta('0 days 00:01:12'), '23']], index=Index([0]), columns=['a', 'b', 'c'])
    tm.assert_frame_equal(result, expected)