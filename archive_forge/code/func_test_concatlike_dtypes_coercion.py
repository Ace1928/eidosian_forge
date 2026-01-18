import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concatlike_dtypes_coercion(self, item, item2, request):
    typ1, vals1 = item
    typ2, vals2 = item2
    vals3 = vals2
    exp_index_dtype = None
    exp_series_dtype = None
    if typ1 == typ2:
        pytest.skip('same dtype is tested in test_concatlike_same_dtypes')
    elif typ1 == 'category' or typ2 == 'category':
        pytest.skip('categorical type tested elsewhere')
    if typ1 == 'bool' and typ2 in ('int64', 'float64'):
        exp_series_dtype = typ2
        mark = pytest.mark.xfail(reason='GH#39187 casting to object')
        request.applymarker(mark)
    elif typ2 == 'bool' and typ1 in ('int64', 'float64'):
        exp_series_dtype = typ1
        mark = pytest.mark.xfail(reason='GH#39187 casting to object')
        request.applymarker(mark)
    elif typ1 in {'datetime64[ns, US/Eastern]', 'timedelta64[ns]'} or typ2 in {'datetime64[ns, US/Eastern]', 'timedelta64[ns]'}:
        exp_index_dtype = object
        exp_series_dtype = object
    exp_data = vals1 + vals2
    exp_data3 = vals1 + vals2 + vals3
    res = Index(vals1).append(Index(vals2))
    exp = Index(exp_data, dtype=exp_index_dtype)
    tm.assert_index_equal(res, exp)
    res = Index(vals1).append([Index(vals2), Index(vals3)])
    exp = Index(exp_data3, dtype=exp_index_dtype)
    tm.assert_index_equal(res, exp)
    res = Series(vals1)._append(Series(vals2), ignore_index=True)
    exp = Series(exp_data, dtype=exp_series_dtype)
    tm.assert_series_equal(res, exp, check_index_type=True)
    res = pd.concat([Series(vals1), Series(vals2)], ignore_index=True)
    tm.assert_series_equal(res, exp, check_index_type=True)
    res = Series(vals1)._append([Series(vals2), Series(vals3)], ignore_index=True)
    exp = Series(exp_data3, dtype=exp_series_dtype)
    tm.assert_series_equal(res, exp)
    res = pd.concat([Series(vals1), Series(vals2), Series(vals3)], ignore_index=True)
    tm.assert_series_equal(res, exp)