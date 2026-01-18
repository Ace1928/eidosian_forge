from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
def test_join_list_series(float_frame):
    left = float_frame.A.to_frame()
    right = [float_frame.B, float_frame[['C', 'D']]]
    result = left.join(right)
    tm.assert_frame_equal(result, float_frame)