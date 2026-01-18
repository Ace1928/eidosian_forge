import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('ignore_index', [True, False])
def test_explode_pyarrow_non_list_type(ignore_index):
    pa = pytest.importorskip('pyarrow')
    data = [1, 2, 3]
    ser = pd.Series(data, dtype=pd.ArrowDtype(pa.int64()))
    result = ser.explode(ignore_index=ignore_index)
    expected = pd.Series([1, 2, 3], dtype='int64[pyarrow]', index=[0, 1, 2])
    tm.assert_series_equal(result, expected)