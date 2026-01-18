import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays import PeriodArray
def test_isin_large_series_mixed_dtypes_and_nan(monkeypatch):
    min_isin_comp = 5
    ser = Series([1, 2, np.nan] * min_isin_comp)
    with monkeypatch.context() as m:
        m.setattr(algorithms, '_MINIMUM_COMP_ARR_LEN', min_isin_comp)
        result = ser.isin({'foo', 'bar'})
    expected = Series([False] * 3 * min_isin_comp)
    tm.assert_series_equal(result, expected)