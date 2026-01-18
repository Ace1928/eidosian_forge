import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('freq, op, result_data', [('ms', 'min', [0.0] * 10), ('ms', 'mean', [0.0] * 9 + [2.0 / 9]), ('ms', 'max', [0.0] * 9 + [2.0]), ('s', 'min', [0.0] * 10), ('s', 'mean', [0.0] * 9 + [2.0 / 9]), ('s', 'max', [0.0] * 9 + [2.0]), ('min', 'min', [0.0] * 10), ('min', 'mean', [0.0] * 9 + [2.0 / 9]), ('min', 'max', [0.0] * 9 + [2.0]), ('h', 'min', [0.0] * 10), ('h', 'mean', [0.0] * 9 + [2.0 / 9]), ('h', 'max', [0.0] * 9 + [2.0]), ('D', 'min', [0.0] * 10), ('D', 'mean', [0.0] * 9 + [2.0 / 9]), ('D', 'max', [0.0] * 9 + [2.0])])
def test_freqs_ops(self, freq, op, result_data):
    index = date_range(start='2018-1-1 01:00:00', freq=f'1{freq}', periods=10)
    s = Series(data=0, index=index, dtype='float')
    s.iloc[1] = np.nan
    s.iloc[-1] = 2
    result = getattr(s.rolling(window=f'10{freq}'), op)()
    expected = Series(data=result_data, index=index)
    tm.assert_series_equal(result, expected)