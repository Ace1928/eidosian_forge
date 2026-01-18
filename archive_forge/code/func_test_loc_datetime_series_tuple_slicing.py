import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_datetime_series_tuple_slicing():
    date = pd.Timestamp('2000')
    ser = Series(1, index=MultiIndex.from_tuples([('a', date)], names=['a', 'b']), name='c')
    result = ser.loc[:, [date]]
    tm.assert_series_equal(result, ser)