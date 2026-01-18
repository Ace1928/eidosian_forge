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
@pytest.mark.parametrize('func', ['min', 'max'])
def test_min_empty_string_dtype(func):
    pytest.importorskip('pyarrow')
    dtype = 'string[pyarrow_numpy]'
    df = DataFrame({'a': ['a'], 'b': 'a', 'c': 'a'}, dtype=dtype).iloc[:0]
    result = getattr(df.groupby('a'), func)()
    expected = DataFrame(columns=['b', 'c'], dtype=dtype, index=pd.Index([], dtype=dtype, name='a'))
    tm.assert_frame_equal(result, expected)