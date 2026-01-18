from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_methods_nunique(test_frame):
    g = test_frame.groupby('A')
    r = g.resample('2s')
    result = r.B.nunique()
    expected = g.B.apply(lambda x: x.resample('2s').nunique())
    tm.assert_series_equal(result, expected)