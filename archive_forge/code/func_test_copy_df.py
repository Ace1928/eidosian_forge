import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_copy_df(self):
    expected = DataFrame({'a': [1, 2, 3]})
    result = SimpleDataFrameSubClass(expected).copy()
    assert type(result) is DataFrame
    tm.assert_frame_equal(result, expected)