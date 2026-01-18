import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_concat_series_length_one_reversed(self, frame_or_series):
    obj = frame_or_series([100])
    result = concat([obj.iloc[::-1]])
    tm.assert_equal(result, obj)