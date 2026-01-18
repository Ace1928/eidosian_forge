import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_with_exclude_string(self, float_frame):
    df = float_frame.copy()
    expected = float_frame.astype(int)
    df['string'] = 'foo'
    casted = df.astype(int, errors='ignore')
    expected['string'] = 'foo'
    tm.assert_frame_equal(casted, expected)
    df = float_frame.copy()
    expected = float_frame.astype(np.int32)
    df['string'] = 'foo'
    casted = df.astype(np.int32, errors='ignore')
    expected['string'] = 'foo'
    tm.assert_frame_equal(casted, expected)