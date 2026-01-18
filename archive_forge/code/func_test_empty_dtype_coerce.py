import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_empty_dtype_coerce(self):
    df1 = DataFrame(data=[[1, None], [2, None]], columns=['a', 'b'])
    df2 = DataFrame(data=[[3, None], [4, None]], columns=['a', 'b'])
    result = concat([df1, df2])
    expected = df1.dtypes
    tm.assert_series_equal(result.dtypes, expected)