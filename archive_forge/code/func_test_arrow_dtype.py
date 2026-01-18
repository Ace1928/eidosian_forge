import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype,exp_dtype', [('Float32', 'Float64'), ('Int16', 'Int32'), ('float[pyarrow]', 'double[pyarrow]')])
def test_arrow_dtype(dtype, exp_dtype):
    pytest.importorskip('pyarrow')
    cols = ['a', 'b']
    df_a = DataFrame([[1, 2], [3, 4], [5, 6]], columns=cols, dtype='int32')
    df_b = DataFrame([[1, 0], [0, 1]], index=cols, dtype=dtype)
    result = df_a.dot(df_b)
    expected = DataFrame([[1, 2], [3, 4], [5, 6]], dtype=exp_dtype)
    tm.assert_frame_equal(result, expected)