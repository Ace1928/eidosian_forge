from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_attach_name_non_reduction_axis1(float_frame):
    result = float_frame.apply(lambda x: np.repeat(x.name, len(x)), axis=1)
    expected = Series((np.repeat(t[0], len(float_frame.columns)) for t in float_frame.itertuples()))
    expected.index = float_frame.index
    tm.assert_series_equal(result, expected)