from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_non_numpy_dtype():
    df = DataFrame({'dt': date_range('2015-01-01', periods=3, tz='Europe/Brussels')})
    result = df.apply(lambda x: x)
    tm.assert_frame_equal(result, df)
    result = df.apply(lambda x: x + pd.Timedelta('1day'))
    expected = DataFrame({'dt': date_range('2015-01-02', periods=3, tz='Europe/Brussels')})
    tm.assert_frame_equal(result, expected)