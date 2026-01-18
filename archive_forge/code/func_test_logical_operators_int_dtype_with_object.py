from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_logical_operators_int_dtype_with_object(self, using_infer_string):
    s_0123 = Series(range(4), dtype='int64')
    result = s_0123 & Series([False, np.nan, False, False])
    expected = Series([False] * 4)
    tm.assert_series_equal(result, expected)
    s_abNd = Series(['a', 'b', np.nan, 'd'])
    if using_infer_string:
        import pyarrow as pa
        with pytest.raises(pa.lib.ArrowNotImplementedError, match='has no kernel'):
            s_0123 & s_abNd
    else:
        with pytest.raises(TypeError, match="unsupported.* 'int' and 'str'"):
            s_0123 & s_abNd