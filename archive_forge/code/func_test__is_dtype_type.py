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
@pytest.mark.parametrize('input_param,result', [(int, np.dtype(int).type), ('int32', np.int32), (float, np.dtype(float).type), ('float64', np.float64), (np.dtype('float64'), np.float64), (str, np.dtype(str).type), (pd.Series([1, 2], dtype=np.dtype('int16')), np.int16), (pd.Series(['a', 'b'], dtype=object), np.object_), (pd.Index([1, 2], dtype='int64'), np.int64), (pd.Index(['a', 'b'], dtype=object), np.object_), ('category', CategoricalDtypeType), (pd.Categorical(['a', 'b']).dtype, CategoricalDtypeType), (pd.Categorical(['a', 'b']), CategoricalDtypeType), (pd.CategoricalIndex(['a', 'b']).dtype, CategoricalDtypeType), (pd.CategoricalIndex(['a', 'b']), CategoricalDtypeType), (pd.DatetimeIndex([1, 2]), np.datetime64), (pd.DatetimeIndex([1, 2]).dtype, np.datetime64), ('<M8[ns]', np.datetime64), (pd.DatetimeIndex(['2000'], tz='Europe/London'), pd.Timestamp), (pd.DatetimeIndex(['2000'], tz='Europe/London').dtype, pd.Timestamp), ('datetime64[ns, Europe/London]', pd.Timestamp), (PeriodDtype(freq='D'), pd.Period), ('period[D]', pd.Period), (IntervalDtype(), pd.Interval), (None, type(None)), (1, type(None)), (1.2, type(None)), (pd.DataFrame([1, 2]), type(None))])
def test__is_dtype_type(input_param, result):
    assert com._is_dtype_type(input_param, lambda tipo: tipo == result)