from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_agg_multiple_mixed():
    mdf = DataFrame({'A': [1, 2, 3], 'B': [1.0, 2.0, 3.0], 'C': ['foo', 'bar', 'baz']})
    expected = DataFrame({'A': [1, 6], 'B': [1.0, 6.0], 'C': ['bar', 'foobarbaz']}, index=['min', 'sum'])
    result = mdf.agg(['min', 'sum'])
    tm.assert_frame_equal(result, expected)
    result = mdf[['C', 'B', 'A']].agg(['sum', 'min'])
    expected = expected[['C', 'B', 'A']].reindex(['sum', 'min'])
    tm.assert_frame_equal(result, expected)