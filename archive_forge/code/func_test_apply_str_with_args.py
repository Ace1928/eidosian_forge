from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('args, kwargs', [([True], {}), ([], {'numeric_only': True})])
def test_apply_str_with_args(df, args, kwargs):
    gb = df.groupby('A')
    result = gb.apply('sum', *args, **kwargs)
    expected = gb.sum(numeric_only=True)
    tm.assert_frame_equal(result, expected)