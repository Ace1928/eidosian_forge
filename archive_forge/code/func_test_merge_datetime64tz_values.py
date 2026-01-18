from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_datetime64tz_values(self):
    left = DataFrame({'key': [1, 2], 'value': pd.date_range('20151010', periods=2, tz='US/Eastern')})
    right = DataFrame({'key': [2, 3], 'value': pd.date_range('20151011', periods=2, tz='US/Eastern')})
    expected = DataFrame({'key': [1, 2, 3], 'value_x': list(pd.date_range('20151010', periods=2, tz='US/Eastern')) + [pd.NaT], 'value_y': [pd.NaT] + list(pd.date_range('20151011', periods=2, tz='US/Eastern'))})
    result = merge(left, right, on='key', how='outer')
    tm.assert_frame_equal(result, expected)
    assert result['value_x'].dtype == 'datetime64[ns, US/Eastern]'
    assert result['value_y'].dtype == 'datetime64[ns, US/Eastern]'