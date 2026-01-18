from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core.arrays import (
def test_mul_listlike_object(self, tda):
    other = np.arange(len(tda))
    result = tda * other.astype(object)
    expected = TimedeltaArray._simple_new(tda._ndarray * other, dtype=tda.dtype)
    tm.assert_extension_array_equal(result, expected)
    assert result._creso == tda._creso