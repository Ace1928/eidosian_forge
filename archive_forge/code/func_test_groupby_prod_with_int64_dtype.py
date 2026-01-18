import builtins
import datetime as dt
from string import ascii_lowercase
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
def test_groupby_prod_with_int64_dtype():
    data = [[1, 11], [1, 41], [1, 17], [1, 37], [1, 7], [1, 29], [1, 31], [1, 2], [1, 3], [1, 43], [1, 5], [1, 47], [1, 19], [1, 88]]
    df = DataFrame(data, columns=['A', 'B'], dtype='int64')
    result = df.groupby(['A']).prod().reset_index()
    expected = DataFrame({'A': [1], 'B': [180970905912331920]}, dtype='int64')
    tm.assert_frame_equal(result, expected)