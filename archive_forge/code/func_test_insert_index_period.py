from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('insert, coerced_val, coerced_dtype', [(pd.Period('2012-01', freq='M'), '2012-01', 'period[M]'), (pd.Timestamp('2012-01-01'), pd.Timestamp('2012-01-01'), object), (1, 1, object), ('x', 'x', object)])
def test_insert_index_period(self, insert, coerced_val, coerced_dtype):
    obj = pd.PeriodIndex(['2011-01', '2011-02', '2011-03', '2011-04'], freq='M')
    assert obj.dtype == 'period[M]'
    data = [pd.Period('2011-01', freq='M'), coerced_val, pd.Period('2011-02', freq='M'), pd.Period('2011-03', freq='M'), pd.Period('2011-04', freq='M')]
    if isinstance(insert, pd.Period):
        exp = pd.PeriodIndex(data, freq='M')
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)
        self._assert_insert_conversion(obj, str(insert), exp, coerced_dtype)
    else:
        result = obj.insert(0, insert)
        expected = obj.astype(object).insert(0, insert)
        tm.assert_index_equal(result, expected)
        if not isinstance(insert, pd.Timestamp):
            result = obj.insert(0, str(insert))
            expected = obj.astype(object).insert(0, str(insert))
            tm.assert_index_equal(result, expected)