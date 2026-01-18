from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_with_tz():
    index = pd.DatetimeIndex(['2013/01/01 09:00', '2013/01/02 09:00'], name='dt1', tz='US/Pacific')
    columns = pd.DatetimeIndex(['2014/01/01 09:00', '2014/01/02 09:00'], name='dt2', tz='Asia/Tokyo')
    result = MultiIndex.from_arrays([index, columns])
    assert result.names == ['dt1', 'dt2']
    tm.assert_index_equal(result.levels[0], index)
    tm.assert_index_equal(result.levels[1], columns)
    result = MultiIndex.from_arrays([Series(index), Series(columns)])
    assert result.names == ['dt1', 'dt2']
    tm.assert_index_equal(result.levels[0], index)
    tm.assert_index_equal(result.levels[1], columns)