import numpy as np
import pytest
from pandas._libs.tslibs import IncompatibleFrequency
from pandas import (
import pandas._testing as tm
def test_is_copy(self, date_range_frame):
    df = date_range_frame.astype({'A': 'float'})
    N = 50
    df.loc[df.index[15:30], 'A'] = np.nan
    dates = date_range('1/1/1990', periods=N * 3, freq='25s')
    result = df.asof(dates)
    with tm.assert_produces_warning(None):
        result['C'] = 1