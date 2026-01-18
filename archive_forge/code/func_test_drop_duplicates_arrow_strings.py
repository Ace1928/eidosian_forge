import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_duplicates_arrow_strings(self):
    pa = pytest.importorskip('pyarrow')
    ser = Series(['a', 'a'], dtype=pd.ArrowDtype(pa.string()))
    result = ser.drop_duplicates()
    expecetd = Series(['a'], dtype=pd.ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expecetd)