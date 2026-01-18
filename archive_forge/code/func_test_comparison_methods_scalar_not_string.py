import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
def test_comparison_methods_scalar_not_string(comparison_op, dtype):
    op_name = f'__{comparison_op.__name__}__'
    a = pd.array(['a', None, 'c'], dtype=dtype)
    other = 42
    if op_name not in ['__eq__', '__ne__']:
        with pytest.raises(TypeError, match='Invalid comparison|not supported between'):
            getattr(a, op_name)(other)
        return
    result = getattr(a, op_name)(other)
    if dtype.storage == 'pyarrow_numpy':
        expected_data = {'__eq__': [False, False, False], '__ne__': [True, True, True]}[op_name]
        expected = np.array(expected_data)
        tm.assert_numpy_array_equal(result, expected)
    else:
        expected_data = {'__eq__': [False, None, False], '__ne__': [True, None, True]}[op_name]
        expected_dtype = 'boolean[pyarrow]' if dtype.storage == 'pyarrow' else 'boolean'
        expected = pd.array(expected_data, dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)