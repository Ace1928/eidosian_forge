from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_mixed_dtype_corner_indexing():
    df = DataFrame({'A': ['foo'], 'B': [1.0]})
    result = df.apply(lambda x: x['A'], axis=1)
    expected = Series(['foo'], index=[0])
    tm.assert_series_equal(result, expected)
    result = df.apply(lambda x: x['B'], axis=1)
    expected = Series([1.0], index=[0])
    tm.assert_series_equal(result, expected)