from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_dups():
    ser = Series(range(5), index=['A', 'A', 'B', 'C', 'C'], dtype=np.int64)
    expected = Series([3, 4], index=['C', 'C'], dtype=np.int64)
    result = ser['C']
    tm.assert_series_equal(result, expected)