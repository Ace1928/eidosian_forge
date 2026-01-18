from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_astype_nan_to_bool(self):
    ser = Series(np.nan, dtype='object')
    result = ser.astype('bool')
    expected = Series(True, dtype='bool')
    tm.assert_series_equal(result, expected)