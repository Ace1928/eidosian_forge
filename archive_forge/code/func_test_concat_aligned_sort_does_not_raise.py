import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_concat_aligned_sort_does_not_raise(self):
    df = DataFrame({1: [1, 2], 'a': [3, 4]}, columns=[1, 'a'])
    expected = DataFrame({1: [1, 2, 1, 2], 'a': [3, 4, 3, 4]}, columns=[1, 'a'])
    result = pd.concat([df, df], ignore_index=True, sort=True)
    tm.assert_frame_equal(result, expected)