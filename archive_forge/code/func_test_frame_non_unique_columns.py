import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
@pytest.mark.parametrize('orient', ['split', 'values'])
@pytest.mark.parametrize('data', [[['a', 'b'], ['c', 'd']], [[1.5, 2.5], [3.5, 4.5]], [[1, 2.5], [3, 4.5]], [[Timestamp('20130101'), 3.5], [Timestamp('20130102'), 4.5]]])
def test_frame_non_unique_columns(self, orient, data):
    df = DataFrame(data, index=[1, 2], columns=['x', 'x'])
    result = read_json(StringIO(df.to_json(orient=orient)), orient=orient, convert_dates=['x'])
    if orient == 'values':
        expected = DataFrame(data)
        if expected.iloc[:, 0].dtype == 'datetime64[ns]':
            expected.isetitem(0, expected.iloc[:, 0].astype(np.int64) // 1000000)
    elif orient == 'split':
        expected = df
        expected.columns = ['x', 'x.1']
    tm.assert_frame_equal(result, expected)