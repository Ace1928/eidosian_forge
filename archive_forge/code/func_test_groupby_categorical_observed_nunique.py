from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_categorical_observed_nunique():
    df = DataFrame({'a': [1, 2], 'b': [1, 2], 'c': [10, 11]})
    df = df.astype(dtype={'a': 'category', 'b': 'category'})
    result = df.groupby(['a', 'b'], observed=True).nunique()['c']
    expected = Series([1, 1], index=MultiIndex.from_arrays([CategoricalIndex([1, 2], name='a'), CategoricalIndex([1, 2], name='b')]), name='c')
    tm.assert_series_equal(result, expected)