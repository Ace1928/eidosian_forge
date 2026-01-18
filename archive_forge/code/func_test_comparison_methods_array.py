import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
def test_comparison_methods_array(comparison_op, dtype):
    op_name = f'__{comparison_op.__name__}__'
    a = pd.array(['a', None, 'c'], dtype=dtype)
    other = [None, None, 'c']
    result = getattr(a, op_name)(other)
    if dtype.storage == 'pyarrow_numpy':
        if operator.ne == comparison_op:
            expected = np.array([True, True, False])
        else:
            expected = np.array([False, False, False])
            expected[-1] = getattr(other[-1], op_name)(a[-1])
        tm.assert_numpy_array_equal(result, expected)
        result = getattr(a, op_name)(pd.NA)
        if operator.ne == comparison_op:
            expected = np.array([True, True, True])
        else:
            expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(result, expected)
    else:
        expected_dtype = 'boolean[pyarrow]' if dtype.storage == 'pyarrow' else 'boolean'
        expected = np.full(len(a), fill_value=None, dtype='object')
        expected[-1] = getattr(other[-1], op_name)(a[-1])
        expected = pd.array(expected, dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)
        result = getattr(a, op_name)(pd.NA)
        expected = pd.array([None, None, None], dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)