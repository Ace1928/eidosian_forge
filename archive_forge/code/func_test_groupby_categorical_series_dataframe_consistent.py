from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_categorical_series_dataframe_consistent(df_cat):
    expected = df_cat.groupby(['A', 'B'], observed=False)['C'].mean()
    result = df_cat.groupby(['A', 'B'], observed=False).mean()['C']
    tm.assert_series_equal(result, expected)