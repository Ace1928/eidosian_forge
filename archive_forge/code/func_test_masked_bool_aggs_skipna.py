import builtins
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('bool_agg_func', ['any', 'all'])
@pytest.mark.parametrize('dtype', ['Int64', 'Float64', 'boolean'])
@pytest.mark.parametrize('skipna', [True, False])
def test_masked_bool_aggs_skipna(bool_agg_func, dtype, skipna, frame_or_series):
    obj = frame_or_series([pd.NA, 1], dtype=dtype)
    expected_res = True
    if not skipna and bool_agg_func == 'all':
        expected_res = pd.NA
    expected = frame_or_series([expected_res], index=np.array([1]), dtype='boolean')
    result = obj.groupby([1, 1]).agg(bool_agg_func, skipna=skipna)
    tm.assert_equal(result, expected)