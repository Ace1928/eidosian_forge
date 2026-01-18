import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
def test_apply_categorical(by_row, using_infer_string):
    values = pd.Categorical(list('ABBABCD'), categories=list('DCBA'), ordered=True)
    ser = Series(values, name='XX', index=list('abcdefg'))
    if not by_row:
        msg = "Series' object has no attribute 'lower"
        with pytest.raises(AttributeError, match=msg):
            ser.apply(lambda x: x.lower(), by_row=by_row)
        assert ser.apply(lambda x: 'A', by_row=by_row) == 'A'
        return
    result = ser.apply(lambda x: x.lower(), by_row=by_row)
    values = pd.Categorical(list('abbabcd'), categories=list('dcba'), ordered=True)
    exp = Series(values, name='XX', index=list('abcdefg'))
    tm.assert_series_equal(result, exp)
    tm.assert_categorical_equal(result.values, exp.values)
    result = ser.apply(lambda x: 'A')
    exp = Series(['A'] * 7, name='XX', index=list('abcdefg'))
    tm.assert_series_equal(result, exp)
    assert result.dtype == object if not using_infer_string else 'string[pyarrow_numpy]'