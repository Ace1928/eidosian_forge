from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_mixed_datetimelike():
    expected = DataFrame({'A': date_range('20130101', periods=3), 'B': pd.to_timedelta(np.arange(3), unit='s')})
    result = expected.apply(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)