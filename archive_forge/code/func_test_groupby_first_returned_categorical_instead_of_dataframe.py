from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('func', ['first', 'last'])
def test_groupby_first_returned_categorical_instead_of_dataframe(func):
    df = DataFrame({'A': [1997], 'B': Series(['b'], dtype='category').cat.as_ordered()})
    df_grouped = df.groupby('A')['B']
    result = getattr(df_grouped, func)()
    expected = Series(['b'], index=Index([1997], name='A'), name='B', dtype=df['B'].dtype)
    tm.assert_series_equal(result, expected)