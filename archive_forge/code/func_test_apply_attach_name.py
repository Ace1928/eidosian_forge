from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_attach_name(float_frame):
    result = float_frame.apply(lambda x: x.name)
    expected = Series(float_frame.columns, index=float_frame.columns)
    tm.assert_series_equal(result, expected)