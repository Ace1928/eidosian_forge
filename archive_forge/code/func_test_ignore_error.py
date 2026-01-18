import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('errors,exp_data', [('ignore', [1, -3.14, 'apple']), ('coerce', [1, -3.14, np.nan])])
@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_ignore_error(errors, exp_data):
    ser = Series([1, -3.14, 'apple'])
    result = to_numeric(ser, errors=errors)
    expected = Series(exp_data)
    tm.assert_series_equal(result, expected)