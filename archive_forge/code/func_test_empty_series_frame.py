import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_empty_series_frame(setup_path):
    s0 = Series(dtype=object)
    s1 = Series(name='myseries', dtype=object)
    df0 = DataFrame()
    df1 = DataFrame(index=['a', 'b', 'c'])
    df2 = DataFrame(columns=['d', 'e', 'f'])
    _check_roundtrip(s0, tm.assert_series_equal, path=setup_path)
    _check_roundtrip(s1, tm.assert_series_equal, path=setup_path)
    _check_roundtrip(df0, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(df1, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(df2, tm.assert_frame_equal, path=setup_path)