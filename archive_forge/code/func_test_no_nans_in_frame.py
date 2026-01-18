import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_no_nans_in_frame(self, axis):
    df = DataFrame([[1, 2], [3, 4]], columns=pd.RangeIndex(0, 2))
    expected = df.copy()
    result = df.dropna(axis=axis)
    tm.assert_frame_equal(result, expected, check_index_type=True)