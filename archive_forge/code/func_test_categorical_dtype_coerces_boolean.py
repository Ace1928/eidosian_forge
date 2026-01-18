from io import StringIO
import os
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data', ['b\nTrue\nFalse\nNA\nFalse', 'b\ntrue\nfalse\nNA\nfalse', 'b\nTRUE\nFALSE\nNA\nFALSE', 'b\nTrue\nFalse\nNA\nFALSE'])
def test_categorical_dtype_coerces_boolean(all_parsers, data):
    parser = all_parsers
    dtype = {'b': CategoricalDtype([False, True])}
    expected = DataFrame({'b': Categorical([True, False, None, False])})
    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)