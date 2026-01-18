from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
def test_nlargest_nan(self):
    df = pd.DataFrame([np.nan, np.nan, 0, 1, 2, 3])
    result = df.nlargest(5, 0)
    expected = df.sort_values(0, ascending=False).head(5)
    tm.assert_frame_equal(result, expected)