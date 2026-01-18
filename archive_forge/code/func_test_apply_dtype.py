from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('col', [1, 1.0, True, 'a', np.nan])
def test_apply_dtype(col):
    df = DataFrame([[1.0, col]], columns=['a', 'b'])
    result = df.apply(lambda x: x.dtype)
    expected = df.dtypes
    tm.assert_series_equal(result, expected)