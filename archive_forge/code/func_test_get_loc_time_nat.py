from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_get_loc_time_nat(self):
    tic = time(minute=12, second=43, microsecond=145224)
    dti = DatetimeIndex([pd.NaT])
    loc = dti.get_loc(tic)
    expected = np.array([], dtype=np.intp)
    tm.assert_numpy_array_equal(loc, expected)