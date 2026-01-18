from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
@pytest.mark.parametrize('target', [[date(2020, 1, 1), Timestamp('2020-01-02')], [Timestamp('2020-01-01'), date(2020, 1, 2)]])
def test_get_indexer_mixed_dtypes(self, target):
    values = DatetimeIndex([Timestamp('2020-01-01'), Timestamp('2020-01-02')])
    result = values.get_indexer(target)
    expected = np.array([0, 1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)