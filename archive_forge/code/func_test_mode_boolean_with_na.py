from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_mode_boolean_with_na(self):
    ser = Series([True, False, True, pd.NA], dtype='boolean')
    result = ser.mode()
    expected = Series({0: True}, dtype='boolean')
    tm.assert_series_equal(result, expected)