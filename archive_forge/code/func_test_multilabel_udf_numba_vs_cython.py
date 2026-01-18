import numpy as np
import pytest
from pandas.errors import NumbaUtilError
from pandas import (
import pandas._testing as tm
def test_multilabel_udf_numba_vs_cython():
    pytest.importorskip('numba')
    df = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'], 'C': np.random.default_rng(2).standard_normal(8), 'D': np.random.default_rng(2).standard_normal(8)})
    gb = df.groupby(['A', 'B'])
    result = gb.transform(lambda values, index: (values - values.min()) / (values.max() - values.min()), engine='numba')
    expected = gb.transform(lambda x: (x - x.min()) / (x.max() - x.min()), engine='cython')
    tm.assert_frame_equal(result, expected)