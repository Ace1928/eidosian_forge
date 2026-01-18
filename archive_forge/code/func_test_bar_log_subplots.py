import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.xfail(np_version_gte1p24 and is_platform_linux(), reason='Weird rounding problems', strict=False)
def test_bar_log_subplots(self):
    expected = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    ax = DataFrame([Series([200, 300]), Series([300, 500])]).plot.bar(log=True, subplots=True)
    tm.assert_numpy_array_equal(ax[0].yaxis.get_ticklocs(), expected)
    tm.assert_numpy_array_equal(ax[1].yaxis.get_ticklocs(), expected)