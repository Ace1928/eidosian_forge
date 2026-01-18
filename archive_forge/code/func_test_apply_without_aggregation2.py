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
def test_apply_without_aggregation2(_test_series):
    grouped = _test_series.to_frame(name='foo').resample('20min', group_keys=False)
    result = grouped['foo'].apply(lambda x: x)
    tm.assert_series_equal(result, _test_series.rename('foo'))