import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_downcast_noop(self, frame_or_series):
    obj = frame_or_series([1, 2, 3], dtype=np.int64)
    msg = "The 'downcast' keyword in fillna"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = obj.fillna('foo', downcast=np.dtype(np.int32))
    expected = obj.astype(np.int32)
    tm.assert_equal(res, expected)
    obj2 = obj.astype(np.float64)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res2 = obj2.fillna('foo', downcast='infer')
    expected2 = obj
    tm.assert_equal(res2, expected2)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res3 = obj2.fillna('foo', downcast=np.dtype(np.int32))
    tm.assert_equal(res3, expected)