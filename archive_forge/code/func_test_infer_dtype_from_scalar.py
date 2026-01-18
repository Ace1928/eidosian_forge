from datetime import (
import numpy as np
import pytest
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import is_dtype_equal
from pandas import (
@pytest.mark.parametrize('value, expected', [('foo', np.object_), (b'foo', np.object_), (1, np.int64), (1.5, np.float64), (np.datetime64('2016-01-01'), np.dtype('M8[s]')), (Timestamp('20160101'), np.dtype('M8[s]')), (Timestamp('20160101', tz='UTC'), 'datetime64[s, UTC]')])
def test_infer_dtype_from_scalar(value, expected, using_infer_string):
    dtype, _ = infer_dtype_from_scalar(value)
    if using_infer_string and value == 'foo':
        expected = 'string'
    assert is_dtype_equal(dtype, expected)
    with pytest.raises(TypeError, match='must be list-like'):
        infer_dtype_from_array(value)