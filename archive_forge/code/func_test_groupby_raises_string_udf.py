import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('how', ['agg', 'transform'])
def test_groupby_raises_string_udf(how, by, groupby_series, df_with_string_col):
    df = df_with_string_col
    gb = df.groupby(by=by)
    if groupby_series:
        gb = gb['d']

    def func(x):
        raise TypeError('Test error message')
    with pytest.raises(TypeError, match='Test error message'):
        getattr(gb, how)(func)