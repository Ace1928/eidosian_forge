from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('args, kwargs', [((1, 2, 3), {}), ((8, 7, 15), {}), ((1, 2), {}), ((1,), {'b': 2}), ((), {'a': 1, 'b': 2}), ((), {'a': 2, 'b': 1}), ((), {'a': 1, 'b': 2, 'c': 3})])
def test_agg_args_kwargs(axis, args, kwargs):

    def f(x, a, b, c=3):
        return x.sum() + (a + b) / c
    df = DataFrame([[1, 2], [3, 4]])
    if axis == 0:
        expected = Series([5.0, 7.0])
    else:
        expected = Series([4.0, 8.0])
    result = df.agg(f, axis, *args, **kwargs)
    tm.assert_series_equal(result, expected)