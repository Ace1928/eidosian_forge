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
def test_bar_log_no_subplots(self):
    expected = np.array([0.1, 1.0, 10.0, 100])
    df = DataFrame({'A': [3] * 5, 'B': list(range(1, 6))}, index=range(5))
    ax = df.plot.bar(grid=True, log=True)
    tm.assert_numpy_array_equal(ax.yaxis.get_ticklocs(), expected)