import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
def test_apply_box_period():
    vals = [pd.Period('2011-01-01', freq='M'), pd.Period('2011-01-02', freq='M')]
    ser = Series(vals)
    assert ser.dtype == 'Period[M]'
    res = ser.apply(lambda x: f'{type(x).__name__}_{x.freqstr}', by_row='compat')
    exp = Series(['Period_M', 'Period_M'])
    tm.assert_series_equal(res, exp)