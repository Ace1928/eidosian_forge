from __future__ import annotations
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.astype import astype_array
import pandas.core.dtypes.common as com
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
import pandas as pd
import pandas._testing as tm
from pandas.api.types import pandas_dtype
from pandas.arrays import SparseArray
@pytest.mark.parametrize('input_param,result', [(int, np.dtype(int)), ('int32', np.dtype('int32')), (float, np.dtype(float)), ('float64', np.dtype('float64')), (np.dtype('float64'), np.dtype('float64')), (str, np.dtype(str)), (pd.Series([1, 2], dtype=np.dtype('int16')), np.dtype('int16')), (pd.Series(['a', 'b'], dtype=object), np.dtype(object)), (pd.Index([1, 2]), np.dtype('int64')), (pd.Index(['a', 'b'], dtype=object), np.dtype(object)), ('category', 'category'), (pd.Categorical(['a', 'b']).dtype, CategoricalDtype(['a', 'b'])), (pd.Categorical(['a', 'b']), CategoricalDtype(['a', 'b'])), (pd.CategoricalIndex(['a', 'b']).dtype, CategoricalDtype(['a', 'b'])), (pd.CategoricalIndex(['a', 'b']), CategoricalDtype(['a', 'b'])), (CategoricalDtype(), CategoricalDtype()), (pd.DatetimeIndex([1, 2]), np.dtype('=M8[ns]')), (pd.DatetimeIndex([1, 2]).dtype, np.dtype('=M8[ns]')), ('<M8[ns]', np.dtype('<M8[ns]')), ('datetime64[ns, Europe/London]', DatetimeTZDtype('ns', 'Europe/London')), (PeriodDtype(freq='D'), PeriodDtype(freq='D')), ('period[D]', PeriodDtype(freq='D')), (IntervalDtype(), IntervalDtype())])
def test_get_dtype(input_param, result):
    assert com._get_dtype(input_param) == result