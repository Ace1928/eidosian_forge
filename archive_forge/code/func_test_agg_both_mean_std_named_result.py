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
@pytest.mark.parametrize('agg', [{'func': {'A': np.mean, 'B': np.std}}, {'A': ('A', np.mean), 'B': ('B', np.std)}, {'A': NamedAgg('A', np.mean), 'B': NamedAgg('B', np.std)}])
def test_agg_both_mean_std_named_result(cases, a_mean, b_std, agg):
    msg = 'using SeriesGroupBy.[mean|std]'
    expected = pd.concat([a_mean, b_std], axis=1)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = cases.aggregate(**agg)
    tm.assert_frame_equal(result, expected, check_like=True)