import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_int(frame_random_data_integer_multi_index):
    df = frame_random_data_integer_multi_index
    result = df.loc[1]
    expected = df[-3:]
    expected.index = expected.index.droplevel(0)
    tm.assert_frame_equal(result, expected)