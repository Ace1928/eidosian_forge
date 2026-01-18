from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_float_frame(float_frame, engine):
    no_rows = float_frame[:0]
    result = no_rows.apply(lambda x: x.mean(), engine=engine)
    expected = Series(np.nan, index=float_frame.columns)
    tm.assert_series_equal(result, expected)
    no_cols = float_frame.loc[:, []]
    result = no_cols.apply(lambda x: x.mean(), axis=1, engine=engine)
    expected = Series(np.nan, index=float_frame.index)
    tm.assert_series_equal(result, expected)