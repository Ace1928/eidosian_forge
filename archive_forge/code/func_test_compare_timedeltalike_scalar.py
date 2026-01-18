from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('td_scalar', [timedelta(days=1), Timedelta(days=1), Timedelta(days=1).to_timedelta64(), offsets.Hour(24)])
def test_compare_timedeltalike_scalar(self, box_with_array, td_scalar):
    box = box_with_array
    xbox = box if box not in [Index, pd.array] else np.ndarray
    ser = Series([timedelta(days=1), timedelta(days=2)])
    ser = tm.box_expected(ser, box)
    actual = ser > td_scalar
    expected = Series([False, True])
    expected = tm.box_expected(expected, xbox)
    tm.assert_equal(actual, expected)