from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_map_na_ignore(float_frame):
    strlen_frame = float_frame.map(lambda x: len(str(x)))
    float_frame_with_na = float_frame.copy()
    mask = np.random.default_rng(2).integers(0, 2, size=float_frame.shape, dtype=bool)
    float_frame_with_na[mask] = pd.NA
    strlen_frame_na_ignore = float_frame_with_na.map(lambda x: len(str(x)), na_action='ignore')
    strlen_frame_with_na = strlen_frame.copy().astype('float64')
    strlen_frame_with_na[mask] = pd.NA
    tm.assert_frame_equal(strlen_frame_na_ignore, strlen_frame_with_na)