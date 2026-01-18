import datetime
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_convert_dtypes_pyarrow_to_np_nullable(self):
    pytest.importorskip('pyarrow')
    ser = pd.DataFrame(range(2), dtype='int32[pyarrow]')
    result = ser.convert_dtypes(dtype_backend='numpy_nullable')
    expected = pd.DataFrame(range(2), dtype='Int32')
    tm.assert_frame_equal(result, expected)