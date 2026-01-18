import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_str_float(self):
    result = DataFrame([np.nan]).astype(str)
    expected = DataFrame(['nan'], dtype='object')
    tm.assert_frame_equal(result, expected)
    result = DataFrame([1.1234567890123457]).astype(str)
    val = '1.1234567890123457'
    expected = DataFrame([val], dtype='object')
    tm.assert_frame_equal(result, expected)