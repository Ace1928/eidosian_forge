from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
def test_join_period_index(frame_with_period_index):
    other = frame_with_period_index.rename(columns=lambda key: f'{key}{key}')
    joined_values = np.concatenate([frame_with_period_index.values] * 2, axis=1)
    joined_cols = frame_with_period_index.columns.append(other.columns)
    joined = frame_with_period_index.join(other)
    expected = DataFrame(data=joined_values, columns=joined_cols, index=frame_with_period_index.index)
    tm.assert_frame_equal(joined, expected)