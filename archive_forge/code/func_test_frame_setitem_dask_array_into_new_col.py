import array
import subprocess
import sys
import numpy as np
import pytest
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_frame_setitem_dask_array_into_new_col():
    olduse = pd.get_option('compute.use_numexpr')
    try:
        da = pytest.importorskip('dask.array')
        dda = da.array([1, 2])
        df = DataFrame({'a': ['a', 'b']})
        df['b'] = dda
        df['c'] = dda
        df.loc[[False, True], 'b'] = 100
        result = df.loc[[1], :]
        expected = DataFrame({'a': ['b'], 'b': [100], 'c': [2]}, index=[1])
        tm.assert_frame_equal(result, expected)
    finally:
        pd.set_option('compute.use_numexpr', olduse)