import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_changing_dtypes_with_duplicate_columns(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), columns=['that', 'that'])
    expected = DataFrame(1.0, index=range(5), columns=['that', 'that'])
    df['that'] = 1.0
    tm.assert_frame_equal(df, expected)
    df = DataFrame(np.random.default_rng(2).random((5, 2)), columns=['that', 'that'])
    expected = DataFrame(1, index=range(5), columns=['that', 'that'])
    df['that'] = 1
    tm.assert_frame_equal(df, expected)