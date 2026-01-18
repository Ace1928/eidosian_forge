import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_describe_exclude_pa_dtype(self):
    pa = pytest.importorskip('pyarrow')
    df = DataFrame({'a': Series([1, 2, 3], dtype=pd.ArrowDtype(pa.int8())), 'b': Series([1, 2, 3], dtype=pd.ArrowDtype(pa.int16())), 'c': Series([1, 2, 3], dtype=pd.ArrowDtype(pa.int32()))})
    result = df.describe(include=pd.ArrowDtype(pa.int8()), exclude=pd.ArrowDtype(pa.int32()))
    expected = DataFrame({'a': [3, 2, 1, 1, 1.5, 2, 2.5, 3]}, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], dtype=pd.ArrowDtype(pa.float64()))
    tm.assert_frame_equal(result, expected)