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
def test_api_append(tmp_path, setup_path):
    path = tmp_path / setup_path
    df = DataFrame(range(20))
    df.iloc[:10].to_hdf(path, key='df', append=True)
    df.iloc[10:].to_hdf(path, key='df', append=True, format='table')
    tm.assert_frame_equal(read_hdf(path, 'df'), df)
    df.iloc[:10].to_hdf(path, key='df', append=False, format='table')
    df.iloc[10:].to_hdf(path, key='df', append=True)
    tm.assert_frame_equal(read_hdf(path, 'df'), df)