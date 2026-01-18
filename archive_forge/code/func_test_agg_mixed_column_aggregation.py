from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_agg_mixed_column_aggregation(cases, a_mean, a_std, b_mean, b_std, request):
    expected = pd.concat([a_mean, a_std, b_mean, b_std], axis=1)
    expected.columns = pd.MultiIndex.from_product([['A', 'B'], ['mean', 'std']])
    msg = 'using SeriesGroupBy.[mean|std]'
    if 'df_mult' in request.node.callspec.id:
        date_mean = cases['date'].mean()
        date_std = cases['date'].std()
        expected = pd.concat([date_mean, date_std, expected], axis=1)
        expected.columns = pd.MultiIndex.from_product([['date', 'A', 'B'], ['mean', 'std']])
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = cases.aggregate([np.mean, np.std])
    tm.assert_frame_equal(result, expected)