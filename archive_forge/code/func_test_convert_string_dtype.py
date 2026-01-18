from itertools import product
import numpy as np
import pytest
from pandas._libs import lib
import pandas as pd
import pandas._testing as tm
def test_convert_string_dtype(self, nullable_string_dtype):
    df = pd.DataFrame({'A': ['a', 'b', pd.NA], 'B': ['ä', 'ö', 'ü']}, dtype=nullable_string_dtype)
    result = df.convert_dtypes()
    tm.assert_frame_equal(df, result)