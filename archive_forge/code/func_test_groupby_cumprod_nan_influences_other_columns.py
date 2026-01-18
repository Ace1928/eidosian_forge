import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
def test_groupby_cumprod_nan_influences_other_columns():
    df = DataFrame({'a': 1, 'b': [1, np.nan, 2], 'c': [1, 2, 3.0]})
    result = df.groupby('a').cumprod(numeric_only=True, skipna=False)
    expected = DataFrame({'b': [1, np.nan, np.nan], 'c': [1, 2, 6.0]})
    tm.assert_frame_equal(result, expected)