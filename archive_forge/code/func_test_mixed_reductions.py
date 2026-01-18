import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('op, expected', [['sum', Series([4, 4], index=['B', 'C'], dtype='Float64')], ['prod', Series([3, 3], index=['B', 'C'], dtype='Float64')], ['min', Series([1, 1], index=['B', 'C'], dtype='Float64')], ['max', Series([3, 3], index=['B', 'C'], dtype='Float64')], ['mean', Series([2, 2], index=['B', 'C'], dtype='Float64')], ['median', Series([2, 2], index=['B', 'C'], dtype='Float64')], ['var', Series([2, 2], index=['B', 'C'], dtype='Float64')], ['std', Series([2 ** 0.5, 2 ** 0.5], index=['B', 'C'], dtype='Float64')], ['skew', Series([pd.NA, pd.NA], index=['B', 'C'], dtype='Float64')], ['kurt', Series([pd.NA, pd.NA], index=['B', 'C'], dtype='Float64')], ['any', Series([True, True, True], index=['A', 'B', 'C'], dtype='boolean')], ['all', Series([True, True, True], index=['A', 'B', 'C'], dtype='boolean')]])
def test_mixed_reductions(op, expected, using_infer_string):
    if op in ['any', 'all'] and using_infer_string:
        expected = expected.astype('bool')
    df = DataFrame({'A': ['a', 'b', 'b'], 'B': [1, None, 3], 'C': array([1, None, 3], dtype='Int64')})
    result = getattr(df.C, op)()
    tm.assert_equal(result, expected['C'])
    if op in ['any', 'all']:
        result = getattr(df, op)()
    else:
        result = getattr(df, op)(numeric_only=True)
    tm.assert_series_equal(result, expected)