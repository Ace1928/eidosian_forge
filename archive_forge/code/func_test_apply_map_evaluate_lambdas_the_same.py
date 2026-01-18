import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
@pytest.mark.parametrize('func', [str, lambda x: str(x)])
def test_apply_map_evaluate_lambdas_the_same(string_series, func, by_row):
    result = string_series.apply(func, by_row=by_row)
    if by_row:
        expected = string_series.map(func)
        tm.assert_series_equal(result, expected)
    else:
        assert result == str(string_series)