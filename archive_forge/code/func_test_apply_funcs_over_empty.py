from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('func', ['sum', 'prod', 'any', 'all'])
def test_apply_funcs_over_empty(func):
    df = DataFrame(columns=['a', 'b', 'c'])
    result = df.apply(getattr(np, func))
    expected = getattr(df, func)()
    if func in ('sum', 'prod'):
        expected = expected.astype(float)
    tm.assert_series_equal(result, expected)