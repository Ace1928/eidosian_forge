import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_describe_percentiles_integer_idx(self):
    df = DataFrame({'x': [1]})
    pct = np.linspace(0, 1, 10 + 1)
    result = df.describe(percentiles=pct)
    expected = DataFrame({'x': [1.0, 1.0, np.nan, 1.0, *(1.0 for _ in pct), 1.0]}, index=['count', 'mean', 'std', 'min', '0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%', 'max'])
    tm.assert_frame_equal(result, expected)