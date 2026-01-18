import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_assert_series_equal_identical_na(nulls_fixture):
    ser = Series([nulls_fixture])
    tm.assert_series_equal(ser, ser.copy())
    idx = pd.Index(ser)
    tm.assert_index_equal(idx, idx.copy(deep=True))