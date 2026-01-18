import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat import IS64
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_large_series(self, monkeypatch):
    size_cutoff = 20
    with monkeypatch.context():
        monkeypatch.setattr(libindex, '_SIZE_CUTOFF', size_cutoff)
        ser = Series(np.arange(size_cutoff), index=IntervalIndex.from_breaks(np.arange(size_cutoff + 1)))
        result1 = ser.loc[:8]
        result2 = ser.loc[0:8]
        result3 = ser.loc[0:8:1]
    tm.assert_series_equal(result1, result2)
    tm.assert_series_equal(result1, result3)