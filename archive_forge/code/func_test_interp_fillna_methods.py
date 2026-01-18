import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('multiblock', [True, False])
@pytest.mark.parametrize('method', ['ffill', 'bfill', 'pad'])
def test_interp_fillna_methods(self, request, axis, multiblock, method, using_array_manager):
    if using_array_manager and axis in (1, 'columns'):
        td.mark_array_manager_not_yet_implemented(request)
    df = DataFrame({'A': [1.0, 2.0, 3.0, 4.0, np.nan, 5.0], 'B': [2.0, 4.0, 6.0, np.nan, 8.0, 10.0], 'C': [3.0, 6.0, 9.0, np.nan, np.nan, 30.0]})
    if multiblock:
        df['D'] = np.nan
        df['E'] = 1.0
    method2 = method if method != 'pad' else 'ffill'
    expected = getattr(df, method2)(axis=axis)
    msg = f'DataFrame.interpolate with method={method} is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.interpolate(method=method, axis=axis)
    tm.assert_frame_equal(result, expected)