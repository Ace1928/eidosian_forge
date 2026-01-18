import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('args, kwargs, increment', [((), {}, 0), ((), {'a': 1}, 1), ((2, 3), {}, 32), ((1,), {'c': 2}, 201)])
def test_agg_args(args, kwargs, increment):

    def f(x, a=0, b=0, c=0):
        return x + a + 10 * b + 100 * c
    s = Series([1, 2])
    result = s.transform(f, 0, *args, **kwargs)
    expected = s + increment
    tm.assert_series_equal(result, expected)