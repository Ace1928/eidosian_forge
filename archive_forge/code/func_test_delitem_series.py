import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_delitem_series(self, data):
    ser = pd.Series(data, name='data')
    taker = np.arange(len(ser))
    taker = np.delete(taker, 1)
    expected = ser[taker]
    del ser[1]
    tm.assert_series_equal(ser, expected)