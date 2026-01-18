import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['first', 'last', 'nth'])
def test_groupby_last_first_nth_with_none(method, nulls_fixture):
    expected = Series(['y'])
    data = Series([nulls_fixture, nulls_fixture, nulls_fixture, 'y', nulls_fixture], index=[0, 0, 0, 0, 0]).groupby(level=0)
    if method == 'nth':
        result = getattr(data, method)(3)
    else:
        result = getattr(data, method)()
    tm.assert_series_equal(result, expected)