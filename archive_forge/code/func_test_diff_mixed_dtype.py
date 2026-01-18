import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_diff_mixed_dtype(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
    df['A'] = np.array([1, 2, 3, 4, 5], dtype=object)
    result = df.diff()
    assert result[0].dtype == np.float64