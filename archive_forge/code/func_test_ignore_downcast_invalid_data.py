import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_ignore_downcast_invalid_data():
    data = ['foo', 2, 3]
    expected = np.array(data, dtype=object)
    msg = "errors='ignore' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = to_numeric(data, errors='ignore', downcast='unsigned')
    tm.assert_numpy_array_equal(res, expected)