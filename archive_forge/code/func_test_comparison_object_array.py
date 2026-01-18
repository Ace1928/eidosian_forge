from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_comparison_object_array(self):
    td = Timedelta('2 days')
    other = Timedelta('3 hours')
    arr = np.array([other, td], dtype=object)
    res = arr == td
    expected = np.array([False, True], dtype=bool)
    assert (res == expected).all()
    arr = np.array([[other, td], [td, other]], dtype=object)
    res = arr != td
    expected = np.array([[True, False], [False, True]], dtype=bool)
    assert res.shape == expected.shape
    assert (res == expected).all()