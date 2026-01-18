import datetime
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_itertuples(self, float_frame):
    for i, tup in enumerate(float_frame.itertuples()):
        ser = DataFrame._constructor_sliced(tup[1:])
        ser.name = tup[0]
        expected = float_frame.iloc[i, :].reset_index(drop=True)
        tm.assert_series_equal(ser, expected)