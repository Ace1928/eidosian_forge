import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_maintain_original_index():
    df = DataFrame({'a': [1, 0, 0, 1], 'b': [0, 1, 0, 0], 'c': [0, 0, 1, 0]}, index=list('abcd'))
    result = from_dummies(df)
    expected = DataFrame({'': list('abca')}, index=list('abcd'))
    tm.assert_frame_equal(result, expected)