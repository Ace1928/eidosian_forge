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
@pytest.mark.parametrize('agg', [{'func': {'A': np.sum, 'B': lambda x: np.std(x, ddof=1)}}, {'A': ('A', np.sum), 'B': ('B', lambda x: np.std(x, ddof=1))}, {'A': NamedAgg('A', np.sum), 'B': NamedAgg('B', lambda x: np.std(x, ddof=1))}])
def test_agg_with_lambda(cases, agg):
    msg = 'using SeriesGroupBy.sum'
    rcustom = cases['B'].apply(lambda x: np.std(x, ddof=1))
    expected = pd.concat([cases['A'].sum(), rcustom], axis=1)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = cases.agg(**agg)
    tm.assert_frame_equal(result, expected, check_like=True)