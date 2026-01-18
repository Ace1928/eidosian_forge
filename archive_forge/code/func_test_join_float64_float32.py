import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_float64_float32(self):
    a = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), columns=['a', 'b'], dtype=np.float64)
    b = DataFrame(np.random.default_rng(2).standard_normal((10, 1)), columns=['c'], dtype=np.float32)
    joined = a.join(b)
    assert joined.dtypes['a'] == 'float64'
    assert joined.dtypes['b'] == 'float64'
    assert joined.dtypes['c'] == 'float32'
    a = np.random.default_rng(2).integers(0, 5, 100).astype('int64')
    b = np.random.default_rng(2).random(100).astype('float64')
    c = np.random.default_rng(2).random(100).astype('float32')
    df = DataFrame({'a': a, 'b': b, 'c': c})
    xpdf = DataFrame({'a': a, 'b': b, 'c': c})
    s = DataFrame(np.random.default_rng(2).random(5).astype('float32'), columns=['md'])
    rs = df.merge(s, left_on='a', right_index=True)
    assert rs.dtypes['a'] == 'int64'
    assert rs.dtypes['b'] == 'float64'
    assert rs.dtypes['c'] == 'float32'
    assert rs.dtypes['md'] == 'float32'
    xp = xpdf.merge(s, left_on='a', right_index=True)
    tm.assert_frame_equal(rs, xp)