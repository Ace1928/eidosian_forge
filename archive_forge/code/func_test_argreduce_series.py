import inspect
import operator
import numpy as np
import pytest
from pandas._typing import Dtype
from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort
@pytest.mark.parametrize('op_name, skipna, expected', [('idxmax', True, 0), ('idxmin', True, 2), ('argmax', True, 0), ('argmin', True, 2), ('idxmax', False, np.nan), ('idxmin', False, np.nan), ('argmax', False, -1), ('argmin', False, -1)])
def test_argreduce_series(self, data_missing_for_sorting, op_name, skipna, expected):
    warn = None
    msg = 'The behavior of Series.argmax/argmin'
    if op_name.startswith('arg') and expected == -1:
        warn = FutureWarning
    if op_name.startswith('idx') and np.isnan(expected):
        warn = FutureWarning
        msg = f'The behavior of Series.{op_name}'
    ser = pd.Series(data_missing_for_sorting)
    with tm.assert_produces_warning(warn, match=msg):
        result = getattr(ser, op_name)(skipna=skipna)
    tm.assert_almost_equal(result, expected)