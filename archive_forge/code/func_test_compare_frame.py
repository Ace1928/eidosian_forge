import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_compare_frame(self):
    data = ['a', 'b', 2, 'a']
    cat = Categorical(data)
    df = DataFrame(cat)
    result = cat == df.T
    expected = DataFrame([[True, True, True, True]])
    tm.assert_frame_equal(result, expected)
    result = cat[::-1] != df.T
    expected = DataFrame([[False, True, True, False]])
    tm.assert_frame_equal(result, expected)