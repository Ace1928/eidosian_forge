from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core.arrays import (
def test_div_numeric_scalar(self, tda):
    other = 2
    result = tda / other
    expected = TimedeltaArray._simple_new(tda._ndarray / other, dtype=tda.dtype)
    tm.assert_extension_array_equal(result, expected)
    assert result._creso == tda._creso