from itertools import product
import numpy as np
import pytest
from pandas._libs import lib
import pandas as pd
import pandas._testing as tm
def test_convert_dtypes_pyarrow_null(self):
    pa = pytest.importorskip('pyarrow')
    ser = pd.Series([None, None])
    result = ser.convert_dtypes(dtype_backend='pyarrow')
    expected = pd.Series([None, None], dtype=pd.ArrowDtype(pa.null()))
    tm.assert_series_equal(result, expected)