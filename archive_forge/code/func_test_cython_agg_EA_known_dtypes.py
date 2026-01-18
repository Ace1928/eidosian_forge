import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('with_na', [True, False])
@pytest.mark.parametrize('op_name, action', [('sum', 'large_int'), ('var', 'always_float'), ('mean', 'always_float'), ('median', 'always_float'), ('prod', 'large_int'), ('min', 'preserve'), ('max', 'preserve'), ('first', 'preserve'), ('last', 'preserve')])
@pytest.mark.parametrize('data', [pd.array([1, 2, 3, 4], dtype='Int64'), pd.array([1, 2, 3, 4], dtype='Int8'), pd.array([0.1, 0.2, 0.3, 0.4], dtype='Float32'), pd.array([0.1, 0.2, 0.3, 0.4], dtype='Float64'), pd.array([True, True, False, False], dtype='boolean')])
def test_cython_agg_EA_known_dtypes(data, op_name, action, with_na):
    if with_na:
        data[3] = pd.NA
    df = DataFrame({'key': ['a', 'a', 'b', 'b'], 'col': data})
    grouped = df.groupby('key')
    if action == 'always_int':
        expected_dtype = pd.Int64Dtype()
    elif action == 'large_int':
        if is_float_dtype(data.dtype):
            expected_dtype = data.dtype
        elif is_integer_dtype(data.dtype):
            expected_dtype = data.dtype
        else:
            expected_dtype = pd.Int64Dtype()
    elif action == 'always_float':
        if is_float_dtype(data.dtype):
            expected_dtype = data.dtype
        else:
            expected_dtype = pd.Float64Dtype()
    elif action == 'preserve':
        expected_dtype = data.dtype
    result = getattr(grouped, op_name)()
    assert result['col'].dtype == expected_dtype
    result = grouped.aggregate(op_name)
    assert result['col'].dtype == expected_dtype
    result = getattr(grouped['col'], op_name)()
    assert result.dtype == expected_dtype
    result = grouped['col'].aggregate(op_name)
    assert result.dtype == expected_dtype