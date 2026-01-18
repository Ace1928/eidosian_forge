from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core.arrays import (
def test_add_timedeltaarraylike(self, tda):
    tda_nano = tda.astype('m8[ns]')
    expected = tda_nano * 2
    res = tda_nano + tda
    tm.assert_extension_array_equal(res, expected)
    res = tda + tda_nano
    tm.assert_extension_array_equal(res, expected)
    expected = tda_nano * 0
    res = tda - tda_nano
    tm.assert_extension_array_equal(res, expected)
    res = tda_nano - tda
    tm.assert_extension_array_equal(res, expected)