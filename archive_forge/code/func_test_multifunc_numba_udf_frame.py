import numpy as np
import pytest
from pandas.errors import NumbaUtilError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('agg_kwargs,expected_func', [({'func': lambda values, index: values.sum()}, 'sum'), pytest.param({'func': [lambda values, index: values.sum(), lambda values, index: values.min()]}, ['sum', 'min'], marks=pytest.mark.xfail(reason="This doesn't work yet! Fails in nopython pipeline!"))])
def test_multifunc_numba_udf_frame(agg_kwargs, expected_func):
    pytest.importorskip('numba')
    data = DataFrame({0: ['a', 'a', 'b', 'b', 'a'], 1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[0, 1, 2])
    grouped = data.groupby(0)
    result = grouped.agg(**agg_kwargs, engine='numba')
    expected = grouped.agg(expected_func, engine='cython')
    tm.assert_frame_equal(result, expected, check_dtype=False)