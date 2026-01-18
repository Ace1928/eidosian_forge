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
def test_df_axis_param_depr():
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq='D')
    index.name = 'date'
    df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=list('AB'), index=index).T
    warning_msg = 'DataFrame.resample with axis=1 is deprecated.'
    with tm.assert_produces_warning(FutureWarning, match=warning_msg):
        df.resample('ME', axis=1)
    df = df.T
    warning_msg = "The 'axis' keyword in DataFrame.resample is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=warning_msg):
        df.resample('ME', axis=0)