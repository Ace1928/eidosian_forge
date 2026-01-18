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
def test_boolean_mask_nullable_int64(self):
    result = DataFrame({'a': [3, 4], 'b': [5, 6]}).astype({'a': 'int64', 'b': 'Int64'})
    mask = Series(False, index=result.index)
    result.loc[mask, 'a'] = result['a']
    result.loc[mask, 'b'] = result['b']
    expected = DataFrame({'a': [3, 4], 'b': [5, 6]}).astype({'a': 'int64', 'b': 'Int64'})
    tm.assert_frame_equal(result, expected)